#include <gst/gst.h>
#include <opencv2/opencv.hpp>
#include "config.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#include "frame_store.h"
#include "rtsp_worker.h"
#include "detector.h"
#include "tracker_manager.h"
#include "static_box_manager.h"
#include "overlay.h"

// ===================== GLOBAL TUNABLE CONSTANTS =====================
//
// Dynamic box  = временный bbox от детектора / трекера
// Static  box  = пользовательский bbox, созданный кликом мыши
// ===================================================================

#define SHOW_IDS = false;

//------------------------------------------------------------------------------
// Максимально допустимая кратность площади результирующего merged-бокса
// относительно площади САМОГО КРУПНОГО динамического бокса в кластере.
// Значение 3.0 означает, что merged-бокс не может превышать
// площадь largest_bbox * 3.
// Используется для ограничения разрастания merged-целей.
static constexpr float MERGE_MAX_AREA_MULTIPLIER = 3.0f;

// Константа сглаживания мелькания размеров dynamic bbox-ов: (в overlay.cpp или main.cpp)
static constexpr int DYN_SMOOTH_WINDOW = 15;
//------------------------------------------------------------------------------
// Maximum number of simultaneously tracked DYNAMIC boxes.
// Limits tracker memory usage and prevents CPU spikes on noisy scenes.
static constexpr int MAX_TARGETS = 50;


//------------------------------------------------------------------------------
// Pixel difference threshold for motion detection.
// Lower values increase sensitivity; higher values suppress noise.
static constexpr int DET_DIFF_THRESHOLD = 20;


//------------------------------------------------------------------------------
// Minimum area (in pixels) for a motion region to become a DYNAMIC box.
static constexpr int DET_MIN_AREA = 10;


//------------------------------------------------------------------------------
// Morphological kernel size for cleaning the motion mask.
static constexpr int DET_MORPH = 3;


//------------------------------------------------------------------------------
// Downscale factor applied before motion detection.
// 1.0 = full resolution, < 1.0 = faster but less precise.
static constexpr double DET_DOWNSCALE = 1.0;


//------------------------------------------------------------------------------
// IoU threshold for matching DYNAMIC boxes between frames.
static constexpr float TRACK_IOU_TH = 0.25f;


//------------------------------------------------------------------------------
// How many consecutive frames a DYNAMIC box may be missing
// before its track is dropped.
static constexpr int TRACK_MAX_MISSED_FRAMES = 3;


//------------------------------------------------------------------------------
// Maximum number of DYNAMIC boxes merged into one cluster.
static constexpr int MERGE_MAX_BOXES_IN_CLUSTER = 2;


//------------------------------------------------------------------------------
// Minimal IoU at which two DYNAMIC boxes are considered neighbors.
static constexpr float MERGE_NEIGHBOR_IOU_TH = 0.05f;


//------------------------------------------------------------------------------
// Center-distance factor for merging DYNAMIC boxes with weak overlap.
static constexpr float MERGE_CENTER_DIST_FACTOR = 5.5f;


//------------------------------------------------------------------------------
// Global transparency for HUD / overlay elements.
static constexpr float HUD_ALPHA = 0.25f;


//------------------------------------------------------------------------------
// Transparency applied to NON-selected objects when a STATIC box exists.
static constexpr float UNSELECTED_ALPHA_WHEN_SELECTED = 0.3f;


// ===================== STATIC BOX SETTINGS =====================


//------------------------------------------------------------------------------
// Enable automatic rebinding of STATIC box after target loss.
static constexpr bool STATIC_AUTO_REBIND_ON_LOSS = true;


//------------------------------------------------------------------------------
// Time (ms) to wait before forced rebinding to a new DYNAMIC box.
static constexpr int STATIC_REBIND_TIMEOUT_MS = 1200;


//------------------------------------------------------------------------------
// Minimal IoU required to keep STATIC box attached to the same parent.
static constexpr float STATIC_PARENT_IOU_TH = 0.15f;


//------------------------------------------------------------------------------
// Minimal score required to reattach STATIC box before forced rebind.
static constexpr float STATIC_REATTACH_SCORE_TH = 0.20f;


// ===================== RTSP SETTINGS =====================

static const char* RTSP_URL = "rtsp://192.168.144.25:8554/main.264";
static constexpr int RTSP_PROTOCOLS = 1;   // UDP
static constexpr int RTSP_LATENCY_MS = 0;
static constexpr guint64 RTSP_TIMEOUT_US = 2000000;
static constexpr guint64 RTSP_TCP_TIMEOUT_US = 2000000;


// ===================== GLOBALS FOR MOUSE CALLBACK =====================

static static_box_manager* g_static_mgr = nullptr;
static std::vector<cv::Rect2f>* g_dynamic_boxes = nullptr;
static std::vector<int>* g_dynamic_ids = nullptr;


// ===================== MOUSE CALLBACK =====================

static void on_mouse(int event, int x, int y, int, void*) {
    if (event != cv::EVENT_LBUTTONDOWN)
        return;

    if (!g_static_mgr || !g_dynamic_boxes || !g_dynamic_ids)
        return;

    g_static_mgr->on_mouse_click(
            x,
            y,
            *g_dynamic_boxes,
            *g_dynamic_ids
    );
}


// ===================== GEOMETRY HELPERS =====================

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


// ===================== MERGE DYNAMIC BOXES =====================

static std::vector<cv::Rect2f>
merge_detections(const std::vector<cv::Rect2f>& dets) {
    std::vector<cv::Rect2f> out;
    if (dets.empty())
        return out;

    std::vector<int> idx(dets.size());
    for (size_t i = 0; i < dets.size(); ++i)
        idx[i] = (int)i;

    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) {
                  return dets[a].area() > dets[b].area();
              });

    std::vector<char> used(dets.size(), 0);

    for (int seed : idx) {
        if (used[seed])
            continue;

        cv::Rect2f merged = dets[seed];
        used[seed] = 1;

        bool grew = true;
        int merged_count = 1;

        while (grew && merged_count < MERGE_MAX_BOXES_IN_CLUSTER) {
            grew = false;
            int best_j = -1;
            float best_score = 0.f;

            for (size_t j = 0; j < dets.size(); ++j) {
                if (used[j])
                    continue;

                float v_iou = iou_rect(merged, dets[j]);
                float d = center_dist(merged, dets[j]);
                float s = ref_size(merged);

                bool neighbor =
                        (v_iou >= MERGE_NEIGHBOR_IOU_TH) ||
                        (d <= MERGE_CENTER_DIST_FACTOR * s);

                if (!neighbor)
                    continue;

                float score = v_iou + 0.001f / (1.0f + d);
                if (score > best_score) {
                    best_score = score;
                    best_j = (int)j;
                }
            }

            if (best_j >= 0) {
                used[best_j] = 1;
                merged |= dets[best_j];
                merged_count++;
                grew = true;
            }
        }

        out.push_back(merged);
    }

    return out;
}


// ===================== MAIN =====================

int main(int argc, char* argv[]) {
    AppConfig cfg;
    load_config("config.toml", cfg);

    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

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

    static_box_manager static_mgr({
                                          STATIC_AUTO_REBIND_ON_LOSS,
                                          STATIC_REBIND_TIMEOUT_MS,
                                          STATIC_PARENT_IOU_TH,
                                          STATIC_REATTACH_SCORE_TH
                                  });

    cv::namedWindow("OPI5", cv::WINDOW_NORMAL);

    std::vector<cv::Rect2f> dynamic_boxes;
    std::vector<int> dynamic_ids;

    g_static_mgr = &static_mgr;
    g_dynamic_boxes = &dynamic_boxes;
    g_dynamic_ids = &dynamic_ids;

    cv::setMouseCallback("OPI5", on_mouse, nullptr);

    // ===================== MAIN LOOP (BLOCKING) =====================
    while (true) {
        cv::Mat frame;

        if (!store.waitFrame(frame, 100)) {
            cv::pollKey();
            continue;
        }

        auto dets = detector.detect(frame);
        auto merged = merge_detections(dets);

        tracker.update(merged);

        dynamic_boxes.clear();
        dynamic_ids.clear();
        for (const auto& t : tracker.targets()) {
            dynamic_boxes.emplace_back(cv::Rect2f(t.bbox));
            dynamic_ids.push_back(t.id);
        }

        static_mgr.update(dynamic_boxes, dynamic_ids);

        overlay.render(frame, tracker.targets(), -1);
        overlay.render_static_boxes(frame, static_mgr.boxes());

        cv::imshow("OPI5", frame);
        cv::pollKey();
    }

    rtsp.stop();
    return 0;
}
