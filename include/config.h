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





