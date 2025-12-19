#include "config.h"

#include <iostream>
#include <toml++/toml.h>

// ============================================================================
// Реализация загрузки config.toml
//
// Важно:
//  - Любая ошибка парсинга не должна "убивать" приложение.
//  - В случае ошибки оставляем дефолты из AppConfig и возвращаем false.
//  - Все имена ключей должны соответствовать config.toml.
// ============================================================================

bool load_config(const std::string& path, AppConfig& cfg) {
    try {
        const auto tbl = toml::parse_file(path);

        // ----------------------------- [rtsp] -----------------------------
        if (auto v = tbl["rtsp"]["url"].value<std::string>())
            cfg.rtsp.url = *v;

        if (auto v = tbl["rtsp"]["protocols"].value<int>())
            cfg.rtsp.protocols = *v;

        if (auto v = tbl["rtsp"]["latency_ms"].value<int>())
            cfg.rtsp.latency_ms = *v;

        if (auto v = tbl["rtsp"]["timeout_us"].value<std::uint64_t>())
            cfg.rtsp.timeout_us = *v;

        if (auto v = tbl["rtsp"]["tcp_timeout_us"].value<std::uint64_t>())
            cfg.rtsp.tcp_timeout_us = *v;

        if (auto v = tbl["rtsp"]["caps_force"].value<std::string>())
            cfg.rtsp.caps_force = *v;

        if (auto v = tbl["rtsp"]["verbose"].value<bool>())
            cfg.rtsp.verbose = *v;


        // ----------------------- [rtsp.watchdog] -----------------------
        // Watchdog перезапуска RTSP, если кадры перестали приходить.
        if (auto v = tbl["rtsp"]["watchdog"]["no_frame_timeout_ms"].value<std::uint64_t>())
            cfg.rtsp_watchdog.no_frame_timeout_ms = *v;

        if (auto v = tbl["rtsp"]["watchdog"]["restart_cooldown_ms"].value<std::uint64_t>())
            cfg.rtsp_watchdog.restart_cooldown_ms = *v;

        if (auto v = tbl["rtsp"]["watchdog"]["startup_grace_ms"].value<std::uint64_t>())
            cfg.rtsp_watchdog.startup_grace_ms = *v;

        // --------------------------- [detector] ---------------------------
        if (auto v = tbl["detector"]["diff_threshold"].value<int>())
            cfg.detector.diff_threshold = *v;

        if (auto v = tbl["detector"]["min_area"].value<int>())
            cfg.detector.min_area = *v;

        if (auto v = tbl["detector"]["morph_kernel"].value<int>())
            cfg.detector.morph_kernel = *v;

        if (auto v = tbl["detector"]["downscale"].value<double>())
            cfg.detector.downscale = *v;

        // ----------------------------- [merge] ----------------------------
        if (auto v = tbl["merge"]["max_boxes_in_cluster"].value<int>())
            cfg.merge.max_boxes_in_cluster = *v;

        if (auto v = tbl["merge"]["neighbor_iou_th"].value<float>())
            cfg.merge.neighbor_iou_th = *v;

        if (auto v = tbl["merge"]["center_dist_factor"].value<float>())
            cfg.merge.center_dist_factor = *v;

        if (auto v = tbl["merge"]["max_area_multiplier"].value<float>())
            cfg.merge.max_area_multiplier = *v;

        // ---------------------------- [tracker] ---------------------------
        if (auto v = tbl["tracker"]["iou_th"].value<float>())
            cfg.tracker.iou_th = *v;

        if (auto v = tbl["tracker"]["max_missed_frames"].value<int>())
            cfg.tracker.max_missed_frames = *v;

        if (auto v = tbl["tracker"]["max_targets"].value<int>())
            cfg.tracker.max_targets = *v;

        // ------------------------- [static_rebind] ------------------------
        if (auto v = tbl["static_rebind"]["auto_rebind"].value<bool>())
            cfg.static_rebind.auto_rebind = *v;

        if (auto v = tbl["static_rebind"]["rebind_timeout_ms"].value<int>())
            cfg.static_rebind.rebind_timeout_ms = *v;

        if (auto v = tbl["static_rebind"]["distance_weight"].value<float>())
            cfg.static_rebind.distance_weight = *v;

        if (auto v = tbl["static_rebind"]["area_weight"].value<float>())
            cfg.static_rebind.area_weight = *v;

        if (auto v = tbl["static_rebind"]["larger_area_factor"].value<float>())
            cfg.static_rebind.larger_area_factor = *v;

        if (auto v = tbl["static_rebind"]["max_large_target_dist_frac"].value<float>())
            cfg.static_rebind.max_large_target_dist_frac = *v;

        if (auto v = tbl["static_rebind"]["parent_iou_th"].value<float>())
            cfg.static_rebind.parent_iou_th = *v;

        if (auto v = tbl["static_rebind"]["reattach_score_th"].value<float>())
            cfg.static_rebind.reattach_score_th = *v;

        // ---------------------------- [overlay] ---------------------------
        if (auto v = tbl["overlay"]["hud_alpha"].value<float>())
            cfg.overlay.hud_alpha = *v;

        if (auto v = tbl["overlay"]["unselected_alpha_when_selected"].value<float>())
            cfg.overlay.unselected_alpha_when_selected = *v;

        // -------------------------- [smoothing] ---------------------------
        if (auto v = tbl["smoothing"]["dynamic_bbox_window"].value<int>())
            cfg.smoothing.dynamic_bbox_window = *v;

        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Config load failed (" << path << "): " << e.what() << std::endl;
        return false;
    }
}
