#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <toml++/toml.h>   // ОБЯЗАТЕЛЬНО, forward-decl НЕЛЬЗЯ


// Читает обязательный ключ из TOML-таблицы и валидирует тип значения.
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



struct OverlayConfig {
    float hud_alpha = 0.25f; // - hud_alpha: прозрачность HUD.

    float unselected_alpha_when_selected = 0.3f; // - unselected_alpha_when_selected: прозрачность невыбранных bbox.
    // -------------------------- [smoothing] ---------------------------
    int dynamic_bbox_window = 5; // - dynamic_bbox_window: окно сглаживания bbox.
};

struct LoggingConfig {
    bool rtsp_level_logger_on = false; // - rtsp_level_logger_on: логировать события RTSP уровня.
    bool manual_detector_level_logger = true; // - manual_detector_level_logger: логировать ручной детектор.
    bool tracker_level_logger = true; // - tracker_level_logger: логировать трекер.
    bool reacquire_level_logger = true; // - reacquire_level_logger: логировать перепривязку.
    bool mouse_click_logger = true; // - mouse_click_logger: логировать клики мыши.
    bool target_object_created_logger = true; // - target_object_created_logger: логировать создание целей.
};

// Загружает конфигурацию логирования из секции [logging].
static inline void load_logging_config(const toml::table& tbl, LoggingConfig& cfg) {
    const auto *logging = tbl["logging"].as_table();
    if (!logging) {
        throw std::runtime_error("missing [logging] table");
    }
    cfg.rtsp_level_logger_on = read_required<bool>(*logging, "rtsp_level_logger_on");
    cfg.manual_detector_level_logger = read_required<bool>(*logging, "manual_detector_level_logger");
    cfg.tracker_level_logger = read_required<bool>(*logging, "tracker_level_logger");
    cfg.reacquire_level_logger = read_required<bool>(*logging, "reacquire_level_logger");
    cfg.mouse_click_logger = read_required<bool>(*logging, "mouse_click_logger");
    cfg.target_object_created_logger = read_required<bool>(*logging, "target_object_created_logger");
}
