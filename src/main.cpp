#include <gst/gst.h>
#include <opencv2/opencv.hpp>

#include <atomic>
#include <chrono>
#include <iostream>
#include <csignal>

#include "core/frame_store.h"
#include "rtsp/rtsp_worker.h"
#include "detect/motion_detector.h"
#include "detect/bbox_filter.h"
#include "tracker/tracker_manager.h"
#include "overlay/overlay_renderer.h"
#include "ui/ui_thread.h"

// ===================== USER TUNABLE =====================

// --- Tracker ---
static constexpr int   MAX_TARGETS = 10;
static constexpr int   TRACKER_UPDATE_EVERY = 1;

// --- Visual tracking performance ---
static constexpr bool   USE_CSRT = true;
static constexpr double OCCLUSION_TIMEOUT_SEC = 1.0;
static constexpr bool   ALLOW_RESYNC = true;
static constexpr int    RESYNC_EVERY_N_FRAMES = 5;

// --- Detection ---
static constexpr float DET_MATCH_IOU   = 0.20f;
static constexpr float SPAWN_BLOCK_IOU = 0.25f;

// --- BBox filter + hysteresis ---
static constexpr int    BBOX_MIN_AREA_PX   = 800;
static constexpr int    BBOX_MAX_AREA_PX   = 120000;
static constexpr double BBOX_MAX_AREA_FRAC = 0.25;

static constexpr int    BBOX_MIN_W = 12;
static constexpr int    BBOX_MIN_H = 12;
static constexpr double BBOX_MIN_AR = 0.15;
static constexpr double BBOX_MAX_AR = 6.0;

static constexpr double BBOX_MATCH_IOU = 0.30;
static constexpr double BBOX_MAX_AREA_JUMP_RATIO = 2.5;
static constexpr int    BBOX_HOLD_MISSED_FRAMES = 3;

// --- UI ---
static constexpr int UI_TARGET_FPS = 30;

// --- Debug ---
static constexpr bool DEBUG_LOGS = true;
static constexpr int  LOG_EVERY_N_FRAMES = 120;

// --- RTSP ---
static const char* RTSP_URL = "rtsp://192.168.144.25:8554/main.264";
static constexpr int     RTSP_PROTOCOLS = 1; // UDP
static constexpr int     RTSP_LATENCY_MS = 0;
static constexpr guint64 RTSP_TIMEOUT_US = 2000000;
static constexpr guint64 RTSP_TCP_TIMEOUT_US = 2000000;

// =======================================================

static std::atomic<int>  g_selected_id{-1};
static std::atomic<bool> g_running{true};
static FrameStore* g_store_ptr = nullptr;

static void onSigInt(int) {
    g_running.store(false, std::memory_order_release);
    if (g_store_ptr)
        g_store_ptr->stop();
}

int main(int argc, char* argv[]) {

    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    gst_init(&argc, &argv);

    FrameStore store;
    g_store_ptr = &store;

    std::signal(SIGINT,  onSigInt);
    std::signal(SIGTERM, onSigInt);

    // ---------------- RTSP ----------------
    RtspWorker::Config rcfg;
    rcfg.url = RTSP_URL;
    rcfg.protocols = RTSP_PROTOCOLS;
    rcfg.latency_ms = RTSP_LATENCY_MS;
    rcfg.timeout_us = RTSP_TIMEOUT_US;
    rcfg.tcp_timeout_us = RTSP_TCP_TIMEOUT_US;

    RtspWorker rtsp(store, rcfg);
    rtsp.start();

    // ---------------- Detector ----------------
    MotionDetector detector(MotionDetector::Config{
            25,   // diff threshold
            250,  // min area
            3,    // morph
            1.0   // downscale
    });

    // ---------------- BBox filter ----------------
    detect::BBoxFilter::Config fcfg;
    fcfg.min_area_px = BBOX_MIN_AREA_PX;
    fcfg.max_area_px = BBOX_MAX_AREA_PX;
    fcfg.max_area_frac = BBOX_MAX_AREA_FRAC;
    fcfg.min_w = BBOX_MIN_W;
    fcfg.min_h = BBOX_MIN_H;
    fcfg.min_ar = BBOX_MIN_AR;
    fcfg.max_ar = BBOX_MAX_AR;
    fcfg.match_iou = BBOX_MATCH_IOU;
    fcfg.max_area_jump_ratio = BBOX_MAX_AREA_JUMP_RATIO;
    fcfg.hold_missed_frames = BBOX_HOLD_MISSED_FRAMES;

    detect::BBoxFilter bbox_filter(fcfg);

    // ---------------- Tracker ----------------
    TrackerManager::Config tcfg;
    tcfg.max_targets = MAX_TARGETS;
    tcfg.use_csrt = USE_CSRT;
    tcfg.occlusion_timeout_sec = OCCLUSION_TIMEOUT_SEC;
    tcfg.allow_resync = ALLOW_RESYNC;
    tcfg.resync_every_n_frames = RESYNC_EVERY_N_FRAMES;
    tcfg.det_match_iou = DET_MATCH_IOU;
    tcfg.spawn_block_iou = SPAWN_BLOCK_IOU;
    tcfg.debug_logs = DEBUG_LOGS;
    tcfg.debug_every_n_frames = LOG_EVERY_N_FRAMES;
    tcfg.tracker_update_every = TRACKER_UPDATE_EVERY;

    TrackerManager tracker(tcfg);

    // ---------------- Overlay ----------------
    OverlayRenderer overlay(OverlayRenderer::Config{
            0.35f,  // alpha
            true    // hide others when selected
    });

    // ---------------- UI thread ----------------
    ui::UiThread uiThread("OPI5", UI_TARGET_FPS);
    uiThread.start();

    int last_click_seq = uiThread.clickState().seq.load(std::memory_order_acquire);

    // ===================== PROCESS LOOP =====================
    // wait/notify, без polling, без waitKey
    // ========================================================

    while (g_running.load(std::memory_order_acquire)) {
        cv::Mat frame;

        if (!store.waitFrame(frame, 2000))
            continue;

        // 1) detect raw
        auto dets = detector.detect(frame);

        // 2) filter + stabilize (no merge)
        auto stable = bbox_filter.process(dets, frame.size());

        // 3) update tracker using stabilized dets
        tracker.update(frame, stable);

        // 4) mouse click handling (thread-safe): UI thread only stores x/y/seq,
        //    processing thread calls tracker->pickTargetId
        const int cur_seq = uiThread.clickState().seq.load(std::memory_order_acquire);
        if (cur_seq != last_click_seq) {
            last_click_seq = cur_seq;
            const int cx = uiThread.clickState().x.load(std::memory_order_acquire);
            const int cy = uiThread.clickState().y.load(std::memory_order_acquire);

            int id = tracker.pickTargetId(cx, cy);
            if (id > 0) {
                g_selected_id.store(id, std::memory_order_release);
                std::cout << "[mouse] selected id=" << id << std::endl;
            }
        }

        // 5) selected target validity
        int sel = g_selected_id.load(std::memory_order_acquire);
        if (sel > 0 && !tracker.hasTargetId(sel)) {
            g_selected_id.store(-1, std::memory_order_release);
            sel = -1;
        }

        // 6) render overlay into frame
        overlay.render(frame, tracker.targets(), sel);

        // 7) send to UI thread (smooth)
        uiThread.submitFrame(frame);
    }

    store.stop();
    rtsp.stop();
    uiThread.stop();
    return 0;
}
