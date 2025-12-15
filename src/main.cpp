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
#include "util/rect_utils.h"

// ===================== TUNABLE CONSTANTS (keep in main, top) =====================

// 1) Max simultaneous targets
static constexpr int MAX_SIMULTANEOUS_TARGETS = 10;

// --- Detector (motion-diff) settings ---
static constexpr int    DET_DIFF_THRESHOLD = 25;
static constexpr int    DET_MIN_AREA       = 250;
static constexpr int    DET_MORPH          = 3;
static constexpr double DET_DOWNSCALE      = 1.0;

// --- Blob merging ---
static constexpr float  MERGE_IOU_THRESHOLD = 0.12f;
static constexpr int    MERGE_GAP_PX        = 10;
static constexpr float  MERGE_EXPAND_FACTOR = 0.08f;

// --- New-target filtering (detector seeds only) ---
static constexpr float  DET_SEED_OVERLAP_IOU = 0.30f; // if overlaps existing track -> NOT a new target

// --- Tracker: occlusion timeout (seconds) ---
static constexpr float  OCCLUSION_TIMEOUT_SEC = 2.0f; // "behind trees" time

// --- Tracker: template tracking (core following) ---
static constexpr bool   ENABLE_TEMPLATE_TRACKING = true;
static constexpr int    TMPL_PATCH_W = 32;
static constexpr int    TMPL_PATCH_H = 32;
static constexpr int    TMPL_SEARCH_PX = 40;
static constexpr float  TMPL_MIN_SCORE = 0.60f;
static constexpr float  TMPL_UPDATE_ALPHA = 0.05f;
static constexpr int    TMPL_MIN_AREA_PX = 200;

// --- Overlay ---
static constexpr float  OVERLAY_ALPHA = 0.35f;
static constexpr bool   HIDE_OTHERS_WHEN_SELECTED = true;

// --- RTSP ---
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

    if (id <= 0) {
        g_selected_id.store(-1, std::memory_order_release);
        std::cout << "[mouse] selection cleared at " << x << "," << y << std::endl;
        return;
    }

    g_selected_id.store(id, std::memory_order_release);
    std::cout << "[mouse] selected id=" << id << " at " << x << "," << y << std::endl;
}

int main(int argc, char *argv[]) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    // keep present but commented:
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

    blob::MergeParams mcfg;
    mcfg.iou_threshold = MERGE_IOU_THRESHOLD;
    mcfg.gap_px        = MERGE_GAP_PX;
    mcfg.expand_factor = MERGE_EXPAND_FACTOR;

    TrackerManager::Config tcfg;
    tcfg.max_targets = MAX_SIMULTANEOUS_TARGETS;
    tcfg.occlusion_timeout_sec = OCCLUSION_TIMEOUT_SEC;
    tcfg.confirm_hits = 2;
    tcfg.seed_overlap_iou = DET_SEED_OVERLAP_IOU;

    tcfg.enable_template_tracking = ENABLE_TEMPLATE_TRACKING;
    tcfg.tmpl_patch_w = TMPL_PATCH_W;
    tcfg.tmpl_patch_h = TMPL_PATCH_H;
    tcfg.tmpl_search_px = TMPL_SEARCH_PX;
    tcfg.tmpl_min_score = TMPL_MIN_SCORE;
    tcfg.tmpl_update_alpha = TMPL_UPDATE_ALPHA;
    tcfg.min_template_area_px = TMPL_MIN_AREA_PX;

    TrackerManager tracker(tcfg);

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

        // Detector sees motion blobs
        auto dets = detector.detect(frame);

        // Merge blobs
        auto merged = blob::merge_blobs(dets, frame.size(), mcfg);

        // Filter: pass ONLY "new targets" into tracker (detector is not allowed to control existing tracks)
        std::vector<cv::Rect2f> new_targets;
        new_targets.reserve(merged.size());

        const auto& existing = tracker.targets();
        for (const auto& d : merged) {
            bool overlaps = false;
            for (const auto& t : existing) {
                if (util::iou(t.bbox, d) >= DET_SEED_OVERLAP_IOU) {
                    overlaps = true;
                    break;
                }
            }
            if (!overlaps) new_targets.push_back(d);
        }

        // Tracker follows all existing targets itself; detections only seed new ones.
        tracker.update(frame, new_targets);

        // Selection auto-reset if disappeared
        int sel = g_selected_id.load(std::memory_order_acquire);
        if (sel > 0) {
            bool sel_exists = false;
            for (const auto& t : tracker.targets()) {
                if (t.id == sel) { sel_exists = true; break; }
            }
            if (!sel_exists) {
                g_selected_id.store(-1, std::memory_order_release);
                sel = -1;
                std::cout << "[ui] selected target disappeared -> selection cleared" << std::endl;
            }
        }

        overlay.render(frame, tracker.targets(), sel);

        cv::imshow("OPI5", frame);
        int k = cv::waitKey(1);
        if (k == 27) break;
    }

    rtsp.stop();
    return 0;
}
