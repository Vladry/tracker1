#include <gst/gst.h>
#include <opencv2/opencv.hpp>

#include <thread>
#include <atomic>
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include "frame_store.h"
#include "rtsp_worker.h"
#include "detector.h"
#include "tracker_manager.h"
#include "overlay.h"

// ===================== TUNABLE CONSTANTS (keep in main, top) =====================

// Max simultaneous targets
static constexpr int MAX_TARGETS = 50;

// Detector settings
static constexpr int DET_DIFF_THRESHOLD = 20;
static constexpr int DET_MIN_AREA = 20;
static constexpr int DET_MORPH = 2;
static constexpr double DET_DOWNSCALE = 2.0;

// Tracker settings (NO KALMAN)
static constexpr float TRACK_IOU_TH = 0.4f;
static constexpr int TRACK_MAX_MISSED_FRAMES = 3;

// Merge / clustering settings
static constexpr int MERGE_MAX_BOXES_IN_CLUSTER = 1;
static constexpr float MERGE_NEIGHBOR_IOU_TH = 0.05f;
static constexpr float MERGE_CENTER_DIST_FACTOR = 5.5f;

// Overlay settings
static constexpr float HUD_ALPHA = 0.25f;
static constexpr float UNSELECTED_ALPHA_WHEN_SELECTED = 0.3f;

// RTSP settings
static const char* RTSP_URL = "rtsp://192.168.144.25:8554/main.264";
static constexpr int RTSP_PROTOCOLS = 1; // UDP
static constexpr int RTSP_LATENCY_MS = 0;
static constexpr guint64 RTSP_TIMEOUT_US = 2000000;
static constexpr guint64 RTSP_TCP_TIMEOUT_US = 2000000;

// ================================================================================

// Selected target id (set by mouse, read by main loop)
static std::atomic<int> g_selected_id{-1};

// ===================== Mouse callback (NON-BLOCKING) =============================

static void on_mouse(int event, int x, int y, int, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN) return;

    auto* tracker = static_cast<TrackerManager*>(userdata);
    if (!tracker) return;

    int id = tracker->pickTargetId(x, y);
    if (id > 0) {
        g_selected_id.store(id, std::memory_order_release);
    }
}

// ===================== Geometry helpers =========================================

static inline float iou_rect(const cv::Rect2f& a, const cv::Rect2f& b) {
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    return (uni > 0.f) ? (inter / uni) : 0.f;
}

static inline float center_dist(const cv::Rect2f& a, const cv::Rect2f& b) {
    float ax = a.x + a.width * 0.5f;
    float ay = a.y + a.height * 0.5f;
    float bx = b.x + b.width * 0.5f;
    float by = b.y + b.height * 0.5f;
    float dx = ax - bx;
    float dy = ay - by;
    return std::sqrt(dx * dx + dy * dy);
}

static inline float ref_size(const cv::Rect2f& r) {
    return std::max(10.0f, 0.5f * (r.width + r.height));
}

// ===================== Merge detections =========================================

static std::vector<cv::Rect2f> merge_detections(const std::vector<cv::Rect2f>& dets) {
    std::vector<cv::Rect2f> out;
    if (dets.empty()) return out;

    std::vector<int> idx(dets.size());
    for (size_t i = 0; i < dets.size(); ++i)
        idx[i] = (int)i;

    std::sort(idx.begin(), idx.end(), [&](int ia, int ib) {
        return dets[(size_t)ia].area() > dets[(size_t)ib].area();
    });

    std::vector<char> used(dets.size(), 0);

    for (int seed_i : idx) {
        if (used[(size_t)seed_i]) continue;

        std::vector<int> cluster;
        cluster.reserve((size_t)MERGE_MAX_BOXES_IN_CLUSTER);
        cluster.push_back(seed_i);
        used[(size_t)seed_i] = 1;

        cv::Rect2f merged = dets[(size_t)seed_i];

        bool grew = true;
        while (grew && (int)cluster.size() < MERGE_MAX_BOXES_IN_CLUSTER) {
            grew = false;

            int best_j = -1;
            float best_score = 0.f;

            for (size_t j = 0; j < dets.size(); ++j) {
                if (used[j]) continue;

                const auto& cand = dets[j];
                float v_iou = iou_rect(merged, cand);
                float d = center_dist(merged, cand);
                float s = ref_size(merged);

                bool is_neighbor =
                        (v_iou >= MERGE_NEIGHBOR_IOU_TH) ||
                        (d <= MERGE_CENTER_DIST_FACTOR * s);

                if (!is_neighbor) continue;

                float score = v_iou + 0.001f * (1.0f / (1.0f + d));
                if (score > best_score) {
                    best_score = score;
                    best_j = (int)j;
                }
            }

            if (best_j >= 0) {
                used[(size_t)best_j] = 1;
                cluster.push_back(best_j);
                merged = merged | dets[(size_t)best_j];
                grew = true;
            }
        }

        out.push_back(merged);
    }

    return out;
}

// ===================== main ======================================================

int main(int argc, char* argv[]) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    // Keep but commented (as agreed):
    // setenv("GST_DEBUG", "rtspsrc:6,rtsp*:6", 1);

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

    MotionDetector detector({
                                    DET_DIFF_THRESHOLD,
                                    DET_MIN_AREA,
                                    DET_MORPH,
                                    DET_DOWNSCALE
                            });

    TrackerManager tracker({
                                   TRACK_IOU_TH,
                                   TRACK_MAX_MISSED_FRAMES,
                                   MAX_TARGETS
                           });

    OverlayRenderer overlay({
                                    HUD_ALPHA,
                                    UNSELECTED_ALPHA_WHEN_SELECTED
                            });

    cv::namedWindow("OPI5", cv::WINDOW_NORMAL);
    cv::setMouseCallback("OPI5", on_mouse, &tracker);

    // ===================== MAIN LOOP (BLOCKING, EVENT-DRIVEN) =====================
    while (true) {
        cv::Mat frame;

        // BLOCK here until a new frame arrives (thread sleeps)
        if (!store.waitFrame(frame, 100)) {
            // allow UI to process mouse events even on timeout
            cv::pollKey();
            continue;
        }

        auto dets = detector.detect(frame);
        auto merged = merge_detections(dets);

        tracker.update(merged);

        // If selected target disappeared — clear selection
        int sel = g_selected_id.load(std::memory_order_acquire);
        if (sel > 0 && !tracker.hasTargetId(sel)) {
            g_selected_id.store(-1, std::memory_order_release);
            sel = -1;
        }

        overlay.render(frame, tracker.targets(), sel);
        cv::imshow("OPI5", frame);

        // Pump UI events (mouse only, keyboard ignored)
        cv::pollKey();
    }

    rtsp.stop();
    return 0;
}
