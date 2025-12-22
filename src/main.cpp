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
#include <atomic>
#include <thread>
#include "display_loop.h"
#include "other_handlers.h"

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

// =====================================================================
// Глобальные флаги управления приложением
//
// ВАЖНО:
//  - выставляются из UI-потока (там где cv::imshow + cv::pollKey)
//  - обрабатываются в control-потоке (watchdog / restart RTSP)
// =====================================================================
std::atomic<bool> g_running{true};
std::atomic<bool> g_rtsp_restart_requested{false};

// Момент времени (в миллисекундах steady_clock) когда мы последний раз
// получили кадр от RTSP. Используется watchdog'ом.
static std::atomic<long long> g_last_frame_ms{0};

// Текущее время по steady_clock в миллисекундах (монотонное, без скачков времени).
static inline long long now_steady_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

static constexpr float MERGE_MAX_AREA_MULTIPLIER = 3.0f;

// Константа сглаживания мелькания размеров dynamic bbox-ов: (в overlay.cpp или main.cpp)
static constexpr int DYN_SMOOTH_WINDOW = 15;
//------------------------------------------------------------------------------
// Maximum number of simultaneously tracked DYNAMIC boxes.
// Limits tracker memory usage and prevents CPU spikes on noisy scenes.
static constexpr int MAX_TARGETS = 50;


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



// ===================== GLOBALS FOR MOUSE CALLBACK =====================

static static_box_manager *g_static_mgr = nullptr;
static std::vector <cv::Rect2f> *g_dynamic_boxes = nullptr;
static std::vector<int> *g_dynamic_ids = nullptr;


// ===================== MOUSE CALLBACK =====================

void on_mouse(int event, int x, int y, int, void *) {
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

static inline float iou_rect(const cv::Rect2f &a, const cv::Rect2f &b) {
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    return (uni > 0.f) ? (inter / uni) : 0.f;
}

static inline float center_dist(const cv::Rect2f &a, const cv::Rect2f &b) {
    float ax = a.x + a.width * 0.5f;
    float ay = a.y + a.height * 0.5f;
    float bx = b.x + b.width * 0.5f;
    float by = b.y + b.height * 0.5f;
    float dx = ax - bx;
    float dy = ay - by;
    return std::sqrt(dx * dx + dy * dy);
}

static inline float ref_size(const cv::Rect2f &r) {
    return std::max(10.0f, 0.5f * (r.width + r.height));
}


// ===================== MERGE DYNAMIC BOXES =====================

static std::vector <cv::Rect2f>
merge_detections(const std::vector <cv::Rect2f> &dets) {
    std::vector <cv::Rect2f> out;
    if (dets.empty())
        return out;

    std::vector<int> idx(dets.size());
    for (size_t i = 0; i < dets.size(); ++i)
        idx[i] = (int) i;

    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) {
                  return dets[a].area() > dets[b].area();
              });

    std::vector<char> used(dets.size(), 0);

    for (int seed: idx) {
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
                    best_j = (int) j;
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

int main(int argc, char *argv[]) {
    AppConfig cfg;
    load_config("config.toml", cfg);

    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    gst_init(&argc, &argv);
    std::cout << "STARTING PIPELINE..." << std::endl;

    FrameStore raw_store;
    FrameStore ui_store;

    RtspWorker::Config rcfg;
    rcfg.url = cfg.rtsp.url;
    rcfg.protocols = cfg.rtsp.protocols;
    rcfg.latency_ms = cfg.rtsp.latency_ms;
    rcfg.timeout_us = cfg.rtsp.timeout_us;
    rcfg.tcp_timeout_us = cfg.rtsp.tcp_timeout_us;
    rcfg.verbose = cfg.rtsp.verbose;

    RtspWorker rtsp(raw_store, rcfg);
    rtsp.start();

    // =====================================================================
    // Control-поток: watchdog RTSP + ручной рестарт по клавише R.
    // Здесь выполняются тяжёлые операции stop/start RTSP, чтобы НЕ блокировать UI.
    // =====================================================================
    static const int WD_NO_FRAME_TIMEOUT_MS = cfg.rtsp_watchdog.no_frame_timeout_ms
                                              ? cfg.rtsp_watchdog.no_frame_timeout_ms : 1500;
    static const int WD_RESTART_COOLDOWN_MS = cfg.rtsp_watchdog.restart_cooldown_ms
                                              ? cfg.rtsp_watchdog.restart_cooldown_ms : 1000;
    static const int WD_STARTUP_GRACE_MS = cfg.rtsp_watchdog.startup_grace_ms ? cfg.rtsp_watchdog.startup_grace_ms
                                                                              : 1500;
    //------------------------------------------------------------------------------
    // Pixel difference threshold for motion detection.
    // Lower values increase sensitivity; higher values suppress noise.
    static const int DET_DIFF_THRESHOLD = cfg.detector.diff_threshold ? cfg.detector.diff_threshold : 20;
    //------------------------------------------------------------------------------
// Minimum area (in pixels) for a motion region to become a DYNAMIC box.
    static const int DET_MIN_AREA = cfg.detector.min_area ? cfg.detector.min_area : 10;
//------------------------------------------------------------------------------
// Morphological kernel size for cleaning the motion mask.
    static const int DET_MORPH = cfg.detector.morph_kernel ? cfg.detector.morph_kernel : 3;
//------------------------------------------------------------------------------
// Downscale factor applied before motion detection.
// 1.0 = full resolution, < 1.0 = faster but less precise.
    static const double DET_DOWNSCALE = cfg.detector.downscale ? cfg.detector.downscale : 1.0;





    const long long app_start_ms = now_steady_ms();
    g_last_frame_ms.store(app_start_ms, std::memory_order_relaxed);

    std::atomic<long long> last_restart_ms{app_start_ms - 100000};

    std::thread control_thread([&]() {
        std::cout << "[CTRL] Control thread started" << std::endl;
        while (g_running.load(std::memory_order_relaxed)) {

            // Ручной рестарт по R / r
            if (g_rtsp_restart_requested.exchange(false, std::memory_order_acq_rel)) {
                std::cout << "[CTRL] Manual RTSP restart" << std::endl;
                rtsp.stop();
                rtsp.start();
                last_restart_ms.store(now_steady_ms(), std::memory_order_relaxed);
                g_last_frame_ms.store(now_steady_ms(), std::memory_order_relaxed);
            }

            // Watchdog: если давно нет кадров — перезапускаем RTSP
            const long long now_ms = now_steady_ms();
            const long long since_start = now_ms - app_start_ms;
            const long long no_frame_ms = now_ms - g_last_frame_ms.load(std::memory_order_relaxed);
            const long long since_restart = now_ms - last_restart_ms.load(std::memory_order_relaxed);

            if (since_start > WD_STARTUP_GRACE_MS &&
                no_frame_ms > WD_NO_FRAME_TIMEOUT_MS &&
                since_restart > WD_RESTART_COOLDOWN_MS) {

                std::cout << "[WATCHDOG] No frames for " << no_frame_ms
                          << " ms -> restarting RTSP" << std::endl;

                rtsp.stop();
                rtsp.start();
                last_restart_ms.store(now_ms, std::memory_order_relaxed);
                g_last_frame_ms.store(now_ms, std::memory_order_relaxed);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::cout << "[CTRL] Control thread exit" << std::endl;
    });


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
    std::vector <cv::Rect2f> dynamic_boxes;
    std::vector<int> dynamic_ids;

    g_static_mgr = &static_mgr;
    g_dynamic_boxes = &dynamic_boxes;
    g_dynamic_ids = &dynamic_ids;

    // ---------------- TRACKER WORKER (из старого while, без UI) ----------------
    std::thread tracker_thread([&]() {
        cv::Mat frame;
        while (g_running.load(std::memory_order_relaxed)) {

            if (!raw_store.waitFrame(frame, 100)) {
                continue;
            }

            // Кадр получен: обновляем таймер watchdog-а.
            g_last_frame_ms.store(now_steady_ms(), std::memory_order_relaxed);

            auto dets = detector.detect(frame);
            auto merged = merge_detections(dets);

            tracker.update(merged);

            dynamic_boxes.clear();
            dynamic_ids.clear();
            for (const auto &t: tracker.targets()) {
                dynamic_boxes.emplace_back(cv::Rect2f(t.bbox));
                dynamic_ids.push_back(t.id);
            }

            static_mgr.update(dynamic_boxes, dynamic_ids);

            overlay.render(frame, tracker.targets(), -1);
            overlay.render_static_boxes(frame, static_mgr.boxes());

            // Публикуем кадр с overlay для UI (отдельный store => нет "саморазгона")
            ui_store.setFrame(std::move(frame));
        }
        std::cout << "[TRK] tracker thread exit" << std::endl;
    });

    DisplayLoop display_loop(ui_store);
    display_loop.run();   // ЕДИНСТВЕННЫЙ UI-LOOP

    if (tracker_thread.joinable()) tracker_thread.join();


// Запрос на остановку RTSP/потоков уже выставлен через g_running.
    // Аккуратно завершаем RTSP.
    rtsp.stop();

    // Будим возможные ожидания по кадрам.
    raw_store.stop();
    ui_store.stop();

    // Дожидаемся control-потока, чтобы он не работал с rtsp после уничтожения.
    if (control_thread.joinable()) control_thread.join();

    return 0;
}