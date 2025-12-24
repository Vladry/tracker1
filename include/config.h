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
struct RtspConfig {
    // RTSP URL камеры. Пример:
    // "rtsp://192.168.144.25:8554/main.264" - для камеры SIYI
    std::string url = "rtsp://192.168.144.25:8554/main.264";

    // Протоколы rtspsrc:
    //(битовая маска) 1 = UDP, 4 = TCP
    int protocols = 1;

    // rtspsrc::latency (мс). 0 = минимальная задержка (но больше риск нестабильности)
    int latency_ms = 0;

    // rtspsrc::timeout / tcp-timeout (микросекунды).
    // Обычно достаточно 2с, чтобы не висеть бесконечно при старте.
    uint64_t timeout_us = 2'000'000;
    uint64_t tcp_timeout_us = 2'000'000;

    // Принудительные caps после декодера (capsfilter).
    // На RK (mppvideodec) типичный стабильный формат для OpenCV: NV12.
    std::string caps_force = "video/x-raw,format=NV12";

    // Таймаут ожидания перехода пайплайна в PLAYING на старте (мс).
    // Если камера/сеть "тупит", лучше явно выйти с ошибкой, чем зависнуть.
    int start_timeout_ms = 3000;

    // Таймаут ожидания перехода в NULL на остановке (мс).
    int stop_timeout_ms = 2000;

    // Пауза после stop() перед повторным стартом (мс).
    // Нужна, чтобы камера/стек RTSP успели закрыть сессию и освободить UDP порты.
    int restart_delay_ms = 300;

    // Подробные логи в stderr.
    bool verbose = true;

};

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
// Static rebind configuration
// ============================================================================

struct StaticRebindConfig {
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

bool load_rtsp_config(toml::table& tbl, RtspConfig& cfg);
bool load_rtsp_watchdog (toml::table &tbl, RtspWatchDog& rtsp_wd);
bool load_merge_config(const toml::table& tbl, MergeConfig& cfg);
bool load_tracker_config(const toml::table& tbl, TrackerConfig& cfg);
bool load_static_rebind_config(const toml::table& tbl, StaticRebindConfig& cfg);
bool load_overlay_config(const toml::table& tbl, OverlayConfig& cfg);
