#include <gst/gst.h>
#include <opencv2/opencv.hpp>

#include <thread>
#include <atomic>
#include <chrono>
#include <iostream>
#include <csignal>

#include "core/frame_store.h"
#include "rtsp/rtsp_worker.h"
#include "detect/motion_detector.h"
#include "tracker/tracker_manager.h"
#include "overlay/overlay_renderer.h"
#include "blob/blob_merger.h"

// ===================== USER TUNABLE =====================

// --- Tracker ---
static constexpr int   MAX_TARGETS = 3;
static constexpr int   TRACKER_UPDATE_EVERY = 3;

// --- Visual tracking performance ---
static constexpr bool  USE_CSRT = true;
static constexpr double OCCLUSION_TIMEOUT_SEC = 1.0;
static constexpr bool  ALLOW_RESYNC = true;
static constexpr int   RESYNC_EVERY_N_FRAMES = 15;

// --- Detection ---
static constexpr float DET_MATCH_IOU   = 0.20f;
static constexpr float SPAWN_BLOCK_IOU = 0.25f;

// --- Debug ---
static constexpr bool  DEBUG_LOGS = true;
static constexpr int   LOG_EVERY_N_FRAMES = 120;

// --- RTSP ---
static const char* RTSP_URL = "rtsp://192.168.144.25:8554/main.264";
static constexpr int RTSP_PROTOCOLS = 1; // UDP
static constexpr int RTSP_LATENCY_MS = 0;
static constexpr guint64 RTSP_TIMEOUT_US = 2000000;
static constexpr guint64 RTSP_TCP_TIMEOUT_US = 2000000;

// =======================================================

static std::atomic<int>  g_selected_id{-1};
static std::atomic<bool> g_running{true};
static FrameStore* g_store_ptr = nullptr;

static void onSigInt(int) {
    g_running.store(false, std::memory_order_release);
    if (g_store_ptr)
        g_store_ptr->stop();   // будим waitFrame()
}

static void onMouse(int event, int x, int y, int, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN) return;
    auto* tracker = static_cast<TrackerManager*>(userdata);
    if (!tracker) return;

    int id = tracker->pickTargetId(x, y);
    if (id > 0) {
        g_selected_id.store(id, std::memory_order_release);
        std::cout << "[mouse] selected id=" << id << std::endl;
    }
}

int main(int argc, char* argv[]) {

    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    gst_init(&argc, &argv);

    FrameStore store;
    g_store_ptr = &store;

    std::signal(SIGINT,  onSigInt);
    std::signal(SIGTERM, onSigInt);

    RtspWorker::Config rcfg;
    rcfg.url = RTSP_URL;
    rcfg.protocols = RTSP_PROTOCOLS;
    rcfg.latency_ms = RTSP_LATENCY_MS;
    rcfg.timeout_us = RTSP_TIMEOUT_US;
    rcfg.tcp_timeout_us = RTSP_TCP_TIMEOUT_US;

    RtspWorker rtsp(store, rcfg);
    rtsp.start();

    MotionDetector detector(MotionDetector::Config{
            25,    // diff threshold
            250,   // min area
            3,     // morph
            1.0    // downscale
    });

    blob::MergeParams mcfg;
    mcfg.iou_threshold = 0.12f;
    mcfg.gap_px = 10;
    mcfg.expand_factor = 0.08f;

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

    OverlayRenderer overlay(OverlayRenderer::Config{
            0.35f,
            true
    });

    cv::namedWindow("OPI5", cv::WINDOW_NORMAL);
    cv::setMouseCallback("OPI5", onMouse, &tracker);

    // ===================== MAIN LOOP =====================
    // НЕТ polling, НЕТ sleep, НЕТ waitKey
    // Поток СПИТ, пока не придёт кадр
    // =====================================================

    while (g_running.load(std::memory_order_acquire)) {
        cv::Mat frame;

        if (!store.waitFrame(frame, 2000)) {
            continue;   // timeout или stop
        }

        auto dets = detector.detect(frame);
        auto merged = blob::merge_blobs(dets, frame.size(), mcfg);

        tracker.update(frame, merged);

        int sel = g_selected_id.load(std::memory_order_acquire);
        if (sel > 0 && !tracker.hasTargetId(sel)) {
            g_selected_id.store(-1, std::memory_order_release);
            sel = -1;
        }

        overlay.render(frame, tracker.targets(), sel);
        cv::imshow("OPI5", frame);

        // ОБЯЗАТЕЛЬНО: прокачка HighGUI без ожиданий
        cv::pollKey();
    }

    store.stop();
    rtsp.stop();
    return 0;
}
