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

// ===================== НАСТРОЕЧНЫЕ ПАРАМЕТРЫ =====================
//
// ВАЖНО: в этом файле не должно быть магических тюнинговых констант.
// Все параметры, влияющие на поведение (пороги, таймауты, веса, лимиты, RTSP),
// находятся в config.toml и загружаются в AppConfig (см. include/config.h).
// ===================================================================

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
merge_detections(const std::vector<cv::Rect2f>& dets, const AppConfig::Merge& mcfg) {
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
        const float seed_area = dets[seed].area();  // площадь самого крупного bbox в кластере (seed)
        used[seed] = 1;

        bool grew = true;
        int merged_count = 1;

        while (grew && merged_count < mcfg.max_boxes_in_cluster) {
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
                        (v_iou >= mcfg.neighbor_iou_th) ||
                        (d <= mcfg.center_dist_factor * s);

                if (!neighbor)
                    continue;

                float score = v_iou + 0.001f / (1.0f + d);
                if (score > best_score) {
                    best_score = score;
                    best_j = (int)j;
                }
            }

            if (best_j >= 0) {
                // Пытаемся расширить merged-бокс, но не даём ему разрастись сверх лимита площади.
                // Это защита от "слипания" множества целей в один огромный bbox.
                const cv::Rect2f tentative = (merged | dets[best_j]);
                if (tentative.area() <= seed_area * mcfg.max_area_multiplier) {
                    used[best_j] = 1;
                    merged = tentative;
                    merged_count++;
                    grew = true;
                }
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
    // ---------------- RTSP (всё берём из config.toml) ----------------
    rcfg.url = cfg.rtsp.url;
    rcfg.protocols = cfg.rtsp.protocols;
    rcfg.latency_ms = cfg.rtsp.latency_ms;
    rcfg.timeout_us = cfg.rtsp.timeout_us;
    rcfg.tcp_timeout_us = cfg.rtsp.tcp_timeout_us;
    rcfg.caps_force = cfg.rtsp.caps_force;
    rcfg.verbose = cfg.rtsp.verbose;

    RtspWorker rtsp(store, rcfg);
    rtsp.start();

    // ---------------- Детектор движения (config.toml -> [detector]) ----------------
    MotionDetector detector(MotionDetector::Config{
            cfg.detector.diff_threshold,
            cfg.detector.min_area,
            cfg.detector.morph_kernel,
            cfg.detector.downscale
    });

    // ---------------- Трекер (config.toml -> [tracker]) ----------------
    TrackerManager tracker(TrackerManager::Config{
            cfg.tracker.iou_th,
            cfg.tracker.max_missed_frames,
            cfg.tracker.max_targets
    });

    // ---------------- Оверлей (config.toml -> [overlay]) ----------------
    OverlayRenderer overlay(OverlayRenderer::Config{
            cfg.overlay.hud_alpha,
            cfg.overlay.unselected_alpha_when_selected
    });

    // ---------------- Static bbox manager (config.toml -> [static_rebind]) ----------------
    static_box_manager static_mgr(static_box_config{
            cfg.static_rebind.auto_rebind,
            cfg.static_rebind.rebind_timeout_ms,
            cfg.static_rebind.parent_iou_th,
            cfg.static_rebind.reattach_score_th
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
        auto merged = merge_detections(dets, cfg.merge);

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
