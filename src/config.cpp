#include <toml++/toml.h>   // ДОЛЖНО БЫТЬ ПЕРВЫМ
#include "config.h"
#include <iostream>

// ============================================================================
// Реализация загрузки config.toml
//
// Важно:
//  - Любая ошибка парсинга не должна "убивать" приложение.
//  - В случае ошибки оставляем дефолты из AppConfig и возвращаем false.
//  - Все имена ключей должны соответствовать config.toml.
// ============================================================================

/*// Старый не нужный, реорганизовать:
bool load_config(const std::string &path, AppConfig &cfg) {
    const auto tbl = toml::parse_file(path);
    const auto tbl = toml::parse_file(path);
    return true;
}*/

bool load_rtsp_config(toml::table &tbl, RtspConfig &cfg) {
    // ----------------------------- [rtsp] -----------------------------
    try {
        const auto& rtsp = tbl.at("rtsp");
        cfg.rtsp.url = rtsp.at("url").value<std::string>().value();
        cfg.rtsp.protocols = rtsp.at("protocols").value<int>().value();
        cfg.rtsp.latency_ms = rtsp.at("latency_ms").value<int>().value();
        cfg.rtsp.timeout_us = rtsp.at("timeout_us").value<std::uint64_t>().value();
        cfg.rtsp.tcp_timeout_us = rtsp.at("tcp_timeout_us").value<std::uint64_t>().value();
        cfg.rtsp.verbose = rtsp.at("verbose").value<bool>().value();
        // ----------------------- [rtsp.watchdog] -----------------------
        const auto &wd = rtsp.at("watchdog");
        // Watchdog перезапуска RTSP, если кадры перестали приходить.
        cfg.rtsp_watchdog.no_frame_timeout_ms = wd.at("no_frame_timeout_ms").value<std::uint64_t>().value();
        cfg.rtsp_watchdog.restart_cooldown_ms = wd.at("restart_cooldown_ms").value<std::uint64_t>().value();
        cfg.rtsp_watchdog.startup_grace_ms = wd.at("startup_grace_ms").value<std::uint64_t>().value();
        return true;

    } catch (const std::exception &e) {
        std::cerr << "rtsp config load failed  " << e.what() << std::endl;
        return false;
    }


};

bool load_detector_config(const toml::table &tbl, DetectorConfig &cfg) {
    // --------------------------- [detector] ---------------------------
    try {
        const auto &detector = tbl.at("detector");
        cfg.detector.diff_threshold = detector("diff_threshold").value<int>().value();
        cfg.detector.min_area = detector("min_area").value<int>().value();
        cfg.detector.morph_kernel = detector("morph_kernel").value<int>().value();
        cfg.detector.downscale = detector("downscale").value<double>().value();
        return true;

    } catch (const std::exception &e) {
        std::cerr << "detecto config load failed  " << e.what() << std::endl;
        return false;
    }

};

bool load_merge_config(const toml::table &tbl, MergeConfig &cfg) {
// ----------------------------- [merge] ----------------------------
    try {
        const auto &merge = tbl.at("merge");
        cfg.merge.max_boxes_in_cluster = merge.at("max_boxes_in_cluster").value<int>().value();
        cfg.merge.neighbor_iou_th = merge.at("neighbor_iou_th").value<float>().value();
        cfg.merge.center_dist_factor = merge.at("center_dist_factor").value<float>().value();
        cfg.merge.max_area_multiplier = merge.at("max_area_multiplier").value<float>().value();

    } catch (const std::exception &e) {
        std::cerr << "merge config load failed  " << e.what() << std::endl;
        return false;
    }
};

bool load_tracker_config(const toml::table &tbl, TrackerConfig &cfg) {
// ---------------------------- [tracker] ---------------------------
    try {
        const auto &tracker = tbl.at("tracker");
        cfg.tracker.iou_th = tracker.at("iou_th").value<float>().value();
        cfg.tracker.max_missed_frames = tracker.at("max_missed_frames").value<int>().value();
        cfg.tracker.max_targets = tracker.at("max_targets").value<int>().value();

    } catch (const std::exception &e) {
        std::cerr << "tracker config load failed  " << e.what() << std::endl;
        return false;
    }
};

bool load_static_rebind_config(const toml::table &tbl, StaticRebindConfig &cfg) {
// ------------------------- [static_rebind] ------------------------
    try {
        const auto &static_rebind = tbl.at("static_rebind");
        cfg.static_rebind.auto_rebind = static_rebind.at("auto_rebind").value<bool>().value();
        cfg.static_rebind.rebind_timeout_ms = static_rebind.at("rebind_timeout_ms").value<int>().value();
        cfg.static_rebind.distance_weight = static_rebind.at("distance_weight").value<float>().value();
        cfg.static_rebind.area_weight = static_rebind.at("area_weight").value<float>().value();
        cfg.static_rebind.larger_area_factor = static_rebind.at("larger_area_factor").value<float>().value();
        cfg.static_rebind.max_large_target_dist_frac = static_rebind.at(
                "max_large_target_dist_frac").value<float>().value();
        cfg.static_rebind.parent_iou_th = static_rebind.at("parent_iou_th").value<float>().value();
        cfg.static_rebind.reattach_score_th = static_rebind.at("reattach_score_th").value<float>().value();

    } catch (const std::exception &e) {
        std::cerr << "static_rebind config load failed  " << e.what() << std::endl;
        return false;
    }
};

bool load_overlay_config(const toml::table &tbl, OverlayConfig &cfg) {
    try {
// ---------------------------- [overlay] ---------------------------
        const auto &overlay = tbl.at("overlay");
        cfg.overlay.hud_alpha = overlay.at("hud_alpha").value<float>().value();
        cfg.overlay.unselected_alpha_when_selected = overlay.at(
                "unselected_alpha_when_selected").value<float>().value();

// -------------------------- [smoothing] ---------------------------
        const auto &smoothing = tbl.at("smoothing");
        cfg.smoothing.dynamic_bbox_window = smoothing.at("dynamic_bbox_window").value<int>().value();

    } catch (const std::exception &e) {
        std::cerr << "overlay config load failed  " << e.what() << std::endl;
        return false;
    }
};

