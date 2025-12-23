#pragma once

#include <string>
#include <cstdint>
#include <toml++/toml.h>   // ОБЯЗАТЕЛЬНО, forward-decl НЕЛЬЗЯ
// ============================================================================
// RTSP / GStreamer configuration
// ============================================================================

struct RtspConfig {

    // ----------------------------- [rtsp] -----------------------------
    struct Rtsp {
        // URL видеопотока (пример: "rtsp://192.168.1.10:8554/main.264")
        std::string url = "";

        // Протоколы rtspsrc:
        // 1 = UDP, 4 = TCP
        int protocols = 1;

        // latency rtspsrc (мс)
        int latency_ms = 0;

        // Таймауты rtspsrc (мкс)
        std::uint64_t timeout_us = 2'000'000;
        std::uint64_t tcp_timeout_us = 2'000'000;

        // Подробные пользовательские логи
        bool verbose = true;
    } rtsp;

    // ------------------------- [rtsp.watchdog] -------------------------
    struct RtspWatchdog {
        // Таймаут отсутствия кадров (мс)
        std::uint64_t no_frame_timeout_ms = 1500;

        // Минимальный интервал между рестартами (мс)
        std::uint64_t restart_cooldown_ms = 1000;

        // Льготный период после старта (мс)
        std::uint64_t startup_grace_ms = 3000;
    } rtsp_watchdog;
};

// ============================================================================
// Detector configuration
// ============================================================================

struct DetectorConfig {

    // --------------------------- [detector] ---------------------------
    struct Detector {
        // Минимальная разница яркости
        int diff_threshold = 20;

        // Минимальная площадь bbox
        int min_area = 10;

        // Размер морфологического ядра
        int morph_kernel = 3;

        // Downscale перед детекцией
        double downscale = 1.0;
    } detector;
};

// ============================================================================
// Merge configuration
// ============================================================================

struct MergeConfig {

    // ----------------------------- [merge] -----------------------------
    struct Merge {
        // Максимум bbox в кластере
        int max_boxes_in_cluster = 2;

        // IoU соседства
        float neighbor_iou_th = 0.05f;

        // Коэффициент расстояния между центрами
        float center_dist_factor = 5.5f;

        // Максимальный рост площади merged bbox
        float max_area_multiplier = 3.0f;
    } merge;
};

// ============================================================================
// Tracker configuration
// ============================================================================

struct TrackerConfig {

    // ---------------------------- [tracker] ----------------------------
    struct Tracker {
        // IoU для сопоставления
        float iou_th = 0.25f;

        // Максимальное число пропущенных кадров
        int max_missed_frames = 3;

        // Максимум активных целей
        int max_targets = 50;
    } tracker;
};

// ============================================================================
// Static rebind configuration
// ============================================================================

struct StaticRebindConfig {

    // ------------------------- [static_rebind] -------------------------
    struct StaticRebind {
        // Автоматическая перепривязка static bbox
        bool auto_rebind = true;

        // Таймаут ожидания новой цели (мс)
        int rebind_timeout_ms = 1200;

        // Вес расстояния
        float distance_weight = 1.0f;

        // Вес площади
        float area_weight = 1.0f;

        // Во сколько раз новая цель должна быть крупнее
        float larger_area_factor = 2.0f;

        // Максимальная допустимая дистанция до крупной цели
        float max_large_target_dist_frac = 0.2f;

        // IoU-порог родительской привязки
        float parent_iou_th = 0.15f;

        // Порог уверенности перепривязки
        float reattach_score_th = 0.20f;
    } static_rebind;
};

// ============================================================================
// Overlay + Smoothing configuration
// ============================================================================

struct OverlayConfig {

    // ---------------------------- [overlay] ----------------------------
    struct Overlay {
        // Прозрачность HUD
        float hud_alpha = 0.25f;

        // Прозрачность невыбранных bbox
        float unselected_alpha_when_selected = 0.3f;
    } overlay;

    // -------------------------- [smoothing] ---------------------------
    struct Smoothing {
        // Окно сглаживания bbox
        int dynamic_bbox_window = 5;
    } smoothing;
};


// ============================================================================
// Loader API (реализовано в config.cpp)
// ============================================================================

bool load_rtsp_config(toml::table& tbl, RtspConfig& cfg);
bool load_detector_config(const toml::table& tbl, DetectorConfig& cfg);
bool load_merge_config(const toml::table& tbl, MergeConfig& cfg);
bool load_tracker_config(const toml::table& tbl, TrackerConfig& cfg);
bool load_static_rebind_config(const toml::table& tbl, StaticRebindConfig& cfg);
bool load_overlay_config(const toml::table& tbl, OverlayConfig& cfg);
