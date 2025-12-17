#pragma once

#include <string>

// ===================== Application Configuration =====================
//
// All configuration values are loaded from config.toml.
// If a parameter is missing, a safe default is used.
// =====================================================================

struct AppConfig {

    struct Detector {
        int diff_threshold = 10;
        int min_area = 20;
        int morph_kernel = 3;
        double downscale = 1.0;
    } detector;

    struct Merge {
        int max_boxes_in_cluster = 3;
        float neighbor_iou_th = 0.05f;
        float center_dist_factor = 5.5f;
        float max_area_multiplier = 3.0f;
    } merge;

    struct Tracker {
        float iou_th = 0.25f;
        int max_missed_frames = 3;
    } tracker;

    struct StaticRebind {
        bool auto_rebind = true;
        int rebind_timeout_ms = 800;
        float distance_weight = 1.0f;
        float area_weight = 1.0f;
        float larger_area_factor = 2.0f;
        float max_large_target_dist_frac = 0.2f;
    } static_rebind;

    struct Overlay {
        float hud_alpha = 0.25f;
        float unselected_alpha_when_selected = 0.3f;
    } overlay;

    struct Smoothing {
        int dynamic_bbox_window = 5;
    } smoothing;
};

// Load configuration from TOML file.
// If file does not exist or contains errors,
// defaults are preserved.
bool load_config(const std::string& path, AppConfig& cfg);
