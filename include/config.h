#pragma once

#include <string>
#include <cstdint>
#include <toml++/toml.h>   // ОБЯЗАТЕЛЬНО, forward-decl НЕЛЬЗЯ


template <typename T>
static T read_required(const toml::table &tbl, std::string_view key) {
    const auto *node = tbl.get(key);
    if (!node) {
        throw std::runtime_error("missing key");
    }
    const auto value = node->value<T>();
    if (!value) {
        throw std::runtime_error("invalid value");
    }
    return *value;
}


// ============================================================================
// RTSP / GStreamer configuration
// ============================================================================


struct RtspWatchDog {
    // ------------------------- [rtsp.watchdog] -------------------------
    // Таймаут отсутствия кадров (мс)
    std::uint64_t no_frame_timeout_ms = 1500;

    // Минимальный интервал между рестартами (мс)
    std::uint64_t restart_cooldown_ms = 1000;

    // Льготный период после старта (мс)
    std::uint64_t startup_grace_ms = 3000;

};

// ============================================================================
// Detector configuration
// ============================================================================


// ============================================================================
// Merge configuration
// ============================================================================
struct MergeConfig {
    // Максимум bbox в кластере
    int max_boxes_in_cluster = 2;

    // IoU соседства
    float neighbor_iou_th = 0.05f;

    // Коэффициент расстояния между центрами
    float center_dist_factor = 5.5f;

    // Максимальный рост площади merged bbox
    float max_area_multiplier = 3.0f;
};




// ============================================================================
// Overlay + Smoothing configuration
// ============================================================================

struct OverlayConfig {
    // Прозрачность HUD
    float hud_alpha = 0.25f;

    // Прозрачность невыбранных bbox
    float unselected_alpha_when_selected = 0.3f;
    // -------------------------- [smoothing] ---------------------------
    // Окно сглаживания bbox
    int dynamic_bbox_window = 5;
};


// ============================================================================
// Loader API (реализовано в config.cpp)
// ============================================================================


bool load_rtsp_watchdog (toml::table &tbl, RtspWatchDog& rtsp_wd);
bool load_merge_config(const toml::table& tbl, MergeConfig& cfg);

