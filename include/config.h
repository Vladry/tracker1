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



struct LoggingConfig {
    bool RTSP_LEVEL_LOGGER_ON = false; // - RTSP_LEVEL_LOGGER_ON: логировать события RTSP уровня.
    bool MANUAL_DETECTOR_LEVEL_LOGGER = true; // - MANUAL_DETECTOR_LEVEL_LOGGER: логировать ручной детектор.
    bool TRACKER_LEVEL_LOGGER = true; // - TRACKER_LEVEL_LOGGER: логировать трекер.
    bool MOUSE_CLICK_LOGGER = true; // - MOUSE_CLICK_LOGGER: логировать клики мыши.
    bool TARGET_OBJECT_CREATED_LOGGER = true; // - TARGET_OBJECT_CREATED_LOGGER: логировать создание целей.
};

// Загружает конфигурацию логирования из секции [logging].
static inline void load_logging_config(const toml::table& tbl, LoggingConfig& cfg) {
    const auto *logging = tbl["logging"].as_table();
    if (!logging) {
        throw std::runtime_error("missing [logging] table");
    }
    cfg.RTSP_LEVEL_LOGGER_ON = read_required<bool>(*logging, "RTSP_LEVEL_LOGGER_ON");
    cfg.MANUAL_DETECTOR_LEVEL_LOGGER = read_required<bool>(*logging, "MANUAL_DETECTOR_LEVEL_LOGGER");
    cfg.TRACKER_LEVEL_LOGGER = read_required<bool>(*logging, "TRACKER_LEVEL_LOGGER");
    cfg.MOUSE_CLICK_LOGGER = read_required<bool>(*logging, "MOUSE_CLICK_LOGGER");
    cfg.TARGET_OBJECT_CREATED_LOGGER = read_required<bool>(*logging, "TARGET_OBJECT_CREATED_LOGGER");
}
