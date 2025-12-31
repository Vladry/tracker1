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



struct OverlayConfig {
    // Прозрачность HUD
    float hud_alpha = 0.25f;

    // Прозрачность невыбранных bbox
    float unselected_alpha_when_selected = 0.3f;
    // -------------------------- [smoothing] ---------------------------
    // Окно сглаживания bbox
    int dynamic_bbox_window = 5;
};

struct LoggingConfig {
    bool rtsp_level_logger_on = false;
    bool manual_detector_level_logger = true;
    bool tracker_level_logger = true;
    bool reacquire_level_logger = true;
    bool mouse_click_logger = true;
    bool target_object_created_logger = true;
};

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
