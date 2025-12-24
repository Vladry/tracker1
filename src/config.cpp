#include <toml++/toml.h>   // ДОЛЖНО БЫТЬ ПЕРВЫМ
#include "config.h"
#include <iostream>
#include <stdexcept>
#include <string_view>


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
        const auto *rtsp_node = tbl.get("rtsp");
        if (!rtsp_node) {
            throw std::runtime_error("missing [rtsp] table");
        }
        const auto *rtsp = rtsp_node->as_table();
        if (!rtsp) {
            throw std::runtime_error("invalid [rtsp] table");
        }
        rcfg.url = read_required<std::string>(*rtsp, "url");
        rcfg.protocols = read_required<int>(*rtsp, "protocols");
        rcfg.latency_ms = read_required<int>(*rtsp, "latency_ms");
        rcfg.timeout_us = read_required<std::uint64_t>(*rtsp, "timeout_us");
        rcfg.tcp_timeout_us = read_required<std::uint64_t>(*rtsp, "tcp_timeout_us");
        rcfg.verbose = read_required<bool>(*rtsp, "verbose");


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
        const auto *rtsp_node = tbl.get("rtsp");
        if (!rtsp_node) {
            throw std::runtime_error("missing [rtsp] table");
        }
        const auto *rtsp = rtsp_node->as_table();
        if (!rtsp) {
            throw std::runtime_error("invalid [rtsp] table");
        }
        const auto *wd_node = rtsp->get("watchdog");
        if (!wd_node) {
            throw std::runtime_error("missing [rtsp.watchdog] table");
        }
        const auto *wd = wd_node->as_table();
        if (!wd) {
            throw std::runtime_error("invalid [rtsp.watchdog] table");
        }
        rtsp_wd.no_frame_timeout_ms = read_required<std::uint64_t>(*wd, "no_frame_timeout_ms");
        rtsp_wd.restart_cooldown_ms = read_required<std::uint64_t>(*wd, "restart_cooldown_ms");
        rtsp_wd.startup_grace_ms = read_required<std::uint64_t>(*wd, "startup_grace_ms");


        return true;

    } catch (const std::exception &e) {
        std::cerr << "rtsp config load failed  " << e.what() << std::endl;
        return false;
    }

};

bool load_merge_config(const toml::table &tbl, MergeConfig &mcfg) {
// ----------------------------- [merge] ----------------------------
    try {
        const auto *merge = tbl["merge"].as_table();
        if (!merge) {
            throw std::runtime_error("missing [merge] table");
        }
        mcfg.max_boxes_in_cluster = read_required<int>(*merge, "max_boxes_in_cluster");
        mcfg.neighbor_iou_th = read_required<float>(*merge, "neighbor_iou_th");
        mcfg.center_dist_factor = read_required<float>(*merge, "center_dist_factor");
        mcfg.max_area_multiplier = read_required<float>(*merge, "max_area_multiplier");
        return true;

    } catch (const std::exception &e) {
        std::cerr << "merge config load failed  " << e.what() << std::endl;
        return false;
    }
};


bool load_static_rebind_config(const toml::table &tbl, StaticRebindConfig &srCfg) {
// ------------------------- [static_rebind] ------------------------
    try {
        const auto *static_rebind = tbl["static_rebind"].as_table();
        if (!static_rebind) {
            throw std::runtime_error("missing [static_rebind] table");
        }
        srCfg.auto_rebind = read_required<bool>(*static_rebind, "auto_rebind");
        srCfg.rebind_timeout_ms = read_required<int>(*static_rebind, "rebind_timeout_ms");
        srCfg.distance_weight = read_required<float>(*static_rebind, "distance_weight");
        srCfg.area_weight = read_required<float>(*static_rebind, "area_weight");
        srCfg.larger_area_factor = read_required<float>(*static_rebind, "larger_area_factor");
        srCfg.max_large_target_dist_frac = read_required<float>(
                *static_rebind, "max_large_target_dist_frac");
        srCfg.parent_iou_th = read_required<float>(*static_rebind, "parent_iou_th");
        srCfg.reattach_score_th = read_required<float>(*static_rebind, "reattach_score_th");
        return true;

    } catch (const std::exception &e) {
        std::cerr << "static_rebind config load failed  " << e.what() << std::endl;
        return false;
    }
};


