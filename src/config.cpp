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
bool load_rtsp_config(toml::table &tbl, RtspConfig &rcfg) {
    // ----------------------------- [rtsp] -----------------------------
    try {
        const auto &rtsp = tbl.at("rtsp");
        rcfg.url = rtsp.at("url").value<std::string>().value();
        rcfg.protocols = rtsp.at("protocols").value<int>().value();
        rcfg.latency_ms = rtsp.at("latency_ms").value<int>().value();
        rcfg.timeout_us = rtsp.at("timeout_us").value<std::uint64_t>().value();
        rcfg.tcp_timeout_us = rtsp.at("tcp_timeout_us").value<std::uint64_t>().value();
        rcfg.verbose = rtsp.at("verbose").value<bool>().value();


        return true;

    } catch (const std::exception &e) {
        std::cerr << "rtsp config load failed  " << e.what() << std::endl;
        return false;
    }
};

bool load_rtsp_watchdog(toml::table &tbl, RtspWatchDog &rtsp_wd) {
    // ----------------------- [rtsp.watchdog] -----------------------
    // Watchdog перезапуска RTSP, если кадры перестали приходить.
    try {
        const auto &wd = tbl.at("rtsp.watchdog");
        rtsp_wd.no_frame_timeout_ms = wd.at("no_frame_timeout_ms").value<std::uint64_t>().value();
        rtsp_wd.restart_cooldown_ms = wd.at("restart_cooldown_ms").value<std::uint64_t>().value();
        rtsp_wd.startup_grace_ms = wd.at("startup_grace_ms").value<std::uint64_t>().value();


        return true;

    } catch (const std::exception &e) {
        std::cerr << "rtsp config load failed  " << e.what() << std::endl;
        return false;
    }

};

bool load_detector_config(const toml::table &tbl, DetectorConfig &dcfg) {
    // --------------------------- [detector] ---------------------------
    try {
        const auto &detector = tbl.at("detector");
        dcfg.diff_threshold = detector.at("diff_threshold").value<int>().value();
        dcfg.min_area = detector.at("min_area").value<int>().value();
        dcfg.morph_kernel = detector.at("morph_kernel").value<int>().value();
        dcfg.downscale = detector.at("downscale").value<double>().value();
        return true;

    } catch (const std::exception &e) {
        std::cerr << "detecto config load failed  " << e.what() << std::endl;
        return false;
    }

};

bool load_merge_config(const toml::table &tbl, MergeConfig &mcfg) {
// ----------------------------- [merge] ----------------------------
    try {
        const auto &merge = tbl.at("merge");
        mcfg.max_boxes_in_cluster = merge.at("max_boxes_in_cluster").value<int>().value();
        mcfg.neighbor_iou_th = merge.at("neighbor_iou_th").value<float>().value();
        mcfg.center_dist_factor = merge.at("center_dist_factor").value<float>().value();
        mcfg.max_area_multiplier = merge.at("max_area_multiplier").value<float>().value();

    } catch (const std::exception &e) {
        std::cerr << "merge config load failed  " << e.what() << std::endl;
        return false;
    }
};

bool load_tracker_config(const toml::table &tbl, TrackerConfig &tcfg) {
// ---------------------------- [tracker] ---------------------------
    try {
        const auto &tracker = tbl.at("tracker");
        tcfg.iou_th = tracker.at("iou_th").value<float>().value();
        tcfg.max_missed_frames = tracker.at("max_missed_frames").value<int>().value();
        tcfg.max_targets = tracker.at("max_targets").value<int>().value();

    } catch (const std::exception &e) {
        std::cerr << "tracker config load failed  " << e.what() << std::endl;
        return false;
    }
};

bool load_static_rebind_config(const toml::table &tbl, StaticRebindConfig &srCfg) {
// ------------------------- [static_rebind] ------------------------
    try {
        const auto &static_rebind = tbl.at("static_rebind");
        srCfg.auto_rebind = static_rebind.at("auto_rebind").value<bool>().value();
        srCfg.rebind_timeout_ms = static_rebind.at("rebind_timeout_ms").value<int>().value();
        srCfg.distance_weight = static_rebind.at("distance_weight").value<float>().value();
        srCfg.area_weight = static_rebind.at("area_weight").value<float>().value();
        srCfg.larger_area_factor = static_rebind.at("larger_area_factor").value<float>().value();
        srCfg.max_large_target_dist_frac = static_rebind.at(
                "max_large_target_dist_frac").value<float>().value();
        srCfg.parent_iou_th = static_rebind.at("parent_iou_th").value<float>().value();
        srCfg.reattach_score_th = static_rebind.at("reattach_score_th").value<float>().value();

    } catch (const std::exception &e) {
        std::cerr << "static_rebind config load failed  " << e.what() << std::endl;
        return false;
    }
};

bool load_overlay_config(const toml::table &tbl, OverlayConfig &ocfg) {
    try {
// ---------------------------- [overlay] ---------------------------
        const auto &overlay = tbl.at("overlay");
        ocfg.hud_alpha = overlay.at("hud_alpha").value<float>().value();
        ocfg.unselected_alpha_when_selected = overlay.at(
                "unselected_alpha_when_selected").value<float>().value();

// -------------------------- [smoothing] ---------------------------
        const auto &smoothing = tbl.at("smoothing");
        ocfg.dynamic_bbox_window = smoothing.at("dynamic_bbox_window").value<int>().value();

    } catch (const std::exception &e) {
        std::cerr << "overlay config load failed  " << e.what() << std::endl;
        return false;
    }
};

