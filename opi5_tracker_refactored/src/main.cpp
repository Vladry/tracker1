#include <gst/gst.h>
#include <opencv2/opencv.hpp>

#include <thread>
#include <atomic>
#include <chrono>
#include <iostream>

#include "core/frame_store.h"
#include "rtsp/rtsp_worker.h"
#include "detect/motion_detector.h"
#include "tracker/tracker_manager.h"
#include "overlay/overlay_renderer.h"
#include "blob/blob_merger.h"

// ===================== TUNABLE CONSTANTS (keep in main, top) =====================

// 1) Max simultaneous targets (global constant as requested)
static constexpr int MAX_SIMULTANEOUS_TARGETS = 10;

// --- Detector (motion-diff) settings ---
static constexpr int    DET_DIFF_THRESHOLD = 25;
static constexpr int    DET_MIN_AREA       = 250;
static constexpr int    DET_MORPH          = 3;
static constexpr double DET_DOWNSCALE      = 1.0;

// --- Blob merging (reduce sub-blobs) ---
static constexpr float  MERGE_IOU_THRESHOLD = 0.12f;
static constexpr int    MERGE_GAP_PX        = 10;
static constexpr float  MERGE_EXPAND_FACTOR = 0.08f;

// --- Tracker association ---
static constexpr float  TRACK_MATCH_IOU_TH      = 0.25f;
static constexpr float  TRACK_ASSOC_IOU_MIN     = 0.05f;
static constexpr float  TRACK_ASSOC_DIST_MAX_PX = 80.0f;
static constexpr float  TRACK_SCORE_IOU_W       = 0.65f;
static constexpr float  TRACK_SCORE_DIST_W      = 0.35f;

// --- Tracker lifecycle / stability ---
static constexpr int    TRACK_CONFIRM_HITS        = 3;
static constexpr int    TRACK_MAX_MISSED_FRAMES   = 30;
static constexpr int    TRACK_STATIONARY_GRACE    = 90;
static constexpr float  TRACK_STATIONARY_SPEED_TH = 0.6f; // px/frame

// --- Kalman tuning ---
static constexpr float  TRACK_PROCESS_NOISE = 1e-2f;
static constexpr float  TRACK_MEAS_NOISE    = 1e-1f;

// --- Overlay settings ---
static constexpr float  OVERLAY_ALPHA = 0.35f;
static constexpr bool   HIDE_OTHERS_WHEN_SELECTED = true;

// --- RTSP settings ---
static const char*      RTSP_URL = "rtsp://192.168.144.25:8554/main.264";
static constexpr int    RTSP_PROTOCOLS = 1; // UDP
static constexpr int    RTSP_LATENCY_MS = 0;
static constexpr guint64 RTSP_TIMEOUT_US = 2000000;
static constexpr guint64 RTSP_TCP_TIMEOUT_US = 2000000;

// ================================================================================

static std::atomic<int> g_selected_id{-1};

static void onMouse(int event, int x, int y, int, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN) return;
    auto* tracker = static_cast<TrackerManager*>(userdata);
    if (!tracker) return;

    int id = tracker->pickTargetId(x, y);
    if (id > 0) {
        g_selected_id.store(id, std::memory_order_release);
        std::cout << "[mouse] selected id=" << id << " at " << x << "," << y << std::endl;
    }
}

int main(int argc, char *argv[]) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    // always keep this line present but commented:
    //    setenv("GST_DEBUG", "rtspsrc:6,rtsp*:6", 1);

    gst_init(&argc, &argv);

    std::cout << "STARTING PIPELINE..." << std::endl;

    FrameStore store;

    RtspWorker::Config rcfg;
    rcfg.url = RTSP_URL;
    rcfg.protocols = RTSP_PROTOCOLS;
    rcfg.latency_ms = RTSP_LATENCY_MS;
    rcfg.timeout_us = RTSP_TIMEOUT_US;
    rcfg.tcp_timeout_us = RTSP_TCP_TIMEOUT_US;

    RtspWorker rtsp(store, rcfg);
    rtsp.start();

    MotionDetector detector(MotionDetector::Config{
        DET_DIFF_THRESHOLD, DET_MIN_AREA, DET_MORPH, DET_DOWNSCALE
    });

    // Blob merger: reduce fragmentation before passing detections into tracker
    blob::MergeParams mcfg;
    mcfg.iou_threshold = MERGE_IOU_THRESHOLD;
    mcfg.gap_px        = MERGE_GAP_PX;
    mcfg.expand_factor = MERGE_EXPAND_FACTOR;

    TrackerManager tracker(TrackerManager::Config{
        TRACK_MATCH_IOU_TH,
        TRACK_ASSOC_IOU_MIN,
        TRACK_ASSOC_DIST_MAX_PX,
        TRACK_SCORE_IOU_W,
        TRACK_SCORE_DIST_W,
        TRACK_CONFIRM_HITS,
        TRACK_MAX_MISSED_FRAMES,
        TRACK_STATIONARY_GRACE,
        TRACK_STATIONARY_SPEED_TH,
        MAX_SIMULTANEOUS_TARGETS,
        TRACK_PROCESS_NOISE,
        TRACK_MEAS_NOISE
    });

    OverlayRenderer overlay(OverlayRenderer::Config{
        OVERLAY_ALPHA, HIDE_OTHERS_WHEN_SELECTED
    });

    cv::namedWindow("OPI5", cv::WINDOW_NORMAL);
    cv::setMouseCallback("OPI5", onMouse, &tracker);

    while (true) {
        cv::Mat frame;
        if (!store.tryGetFrame(frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        auto dets = detector.detect(frame);

        // merge blobs to reduce "one body -> many bboxes"
        auto merged = blob::merge_blobs(dets, frame.size(), mcfg);

        tracker.update(merged);

        int sel = g_selected_id.load(std::memory_order_acquire);
        overlay.render(frame, tracker.targets(), sel);

        cv::imshow("OPI5", frame);
        int k = cv::waitKey(1);
        if (k == 27) break;
    }

    rtsp.stop();
    return 0;
}
