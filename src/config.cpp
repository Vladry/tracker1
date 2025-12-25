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




