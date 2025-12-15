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

// ===================== TUNABLE CONSTANTS =====================

// --- Tracker mode ---
static constexpr bool USE_KALMAN = false;

// --- Limits ---
static constexpr int MAX_SIMULTANEOUS_TARGETS = 10;

// --- Detector ---
static constexpr int    DET_DIFF_THRESHOLD = 25;
static constexpr int    DET_MIN_AREA       = 250;
static constexpr int    DET_MORPH          = 3;
static constexpr double DET_DOWNSCALE      = 1.0;

// --- Blob merge ---
static constexpr float  MERGE_IOU_THRESHOLD = 0.12f;
static constexpr int    MERGE_GAP_PX        = 10;
static constexpr float  MERGE_EXPAND_FACTOR = 0.08f;

// --- Tracker association ---
static constexpr float  ASSOC_IOU_TH      = 0.25f;
static constexpr float  SPAWN_BLOCK_IOU   = 0.30f;

// --- Kalman ---
static constexpr float  KALMAN_PROCESS_NOISE = 1e-2f;
static constexpr float  KALMAN_MEAS_NOISE    = 1e-1f;
static constexpr float  STATIONARY_SPEED_PX  = 0.5f;

// --- Tracker lifecycle ---
static constexpr float  OCCLUSION_TIMEOUT_SEC = 2.0f;
static constexpr float  STATIONARY_HOLD_SEC   = 30.0f;
static constexpr int    TRACK_CONFIRM_HITS    = 2;

// --- Overlay ---
static constexpr float  OVERLAY_ALPHA = 0.35f;
static constexpr bool   HIDE_OTHERS_WHEN_SELECTED = true;

// --- RTSP ---
static const char* RTSP_URL = "rtsp://192.168.144.25:8554/main.264";
static constexpr int RTSP_PROTOCOLS = 1;
static constexpr int RTSP_LATENCY_MS = 0;
static constexpr guint64 RTSP_TIMEOUT_US = 2000000;
static constexpr guint64 RTSP_TCP_TIMEOUT_US = 2000000;

// ============================================================

static std::atomic<int> g_selected_id{-1};

static void onMouse(int event, int x, int y, int, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN) return;
    auto* tracker = static_cast<TrackerManager*>(userdata);
    if (!tracker) return;

    int id = tracker->pickTargetId(x, y);
    if (id >= 0) g_selected_id.store(id);
}

int main(int argc, char *argv[]) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    // keep commented
    // setenv("GST_DEBUG", "rtspsrc:6,rtsp*:6", 1);

    gst_init(&argc, &argv);

    FrameStore store;

    RtspWorker::Config rcfg;
    rcfg.url = RTSP_URL;
    rcfg.protocols = RTSP_PROTOCOLS;
    rcfg.latency_ms = RTSP_LATENCY_MS;
    rcfg.timeout_us = RTSP_TIMEOUT_US;
    rcfg.tcp_timeout_us = RTSP_TCP_TIMEOUT_US;

    RtspWorker rtsp(store, rcfg);
    rtsp.start();

    MotionDetector detector({
                                    DET_DIFF_THRESHOLD,
                                    DET_MIN_AREA,
                                    DET_MORPH,
                                    DET_DOWNSCALE
                            });

    blob::MergeParams mcfg;
    mcfg.iou_threshold = MERGE_IOU_THRESHOLD;
    mcfg.gap_px        = MERGE_GAP_PX;
    mcfg.expand_factor = MERGE_EXPAND_FACTOR;

    TrackerManager::Config tcfg;
    tcfg.max_targets = MAX_SIMULTANEOUS_TARGETS;
    tcfg.occlusion_timeout_sec = OCCLUSION_TIMEOUT_SEC;
    tcfg.stationary_hold_sec   = STATIONARY_HOLD_SEC;
    tcfg.confirm_hits          = TRACK_CONFIRM_HITS;

    tcfg.assoc_iou_threshold = ASSOC_IOU_TH;
    tcfg.spawn_block_iou     = SPAWN_BLOCK_IOU;

    tcfg.use_kalman = USE_KALMAN;
    tcfg.kalman_process_noise = KALMAN_PROCESS_NOISE;
    tcfg.kalman_meas_noise    = KALMAN_MEAS_NOISE;
    tcfg.stationary_speed_px  = STATIONARY_SPEED_PX;

    TrackerManager tracker(tcfg);

    OverlayRenderer overlay({
                                    OVERLAY_ALPHA,
                                    HIDE_OTHERS_WHEN_SELECTED
                            });

    cv::namedWindow("OPI5", cv::WINDOW_NORMAL);
    cv::setMouseCallback("OPI5", onMouse, &tracker);

    while (true) {
        cv::Mat frame;
        if (!store.tryGetFrame(frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        auto dets   = detector.detect(frame);
        auto merged = blob::merge_blobs(dets, frame.size(), mcfg);

        tracker.update(frame, merged);

        int sel = g_selected_id.load();
        overlay.render(frame, tracker.targets(), sel);

        cv::imshow("OPI5", frame);
        if (cv::waitKey(1) == 27) break;
    }

    rtsp.stop();
    return 0;
}
