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





bool RtspWatchDog::load_merge_config(const toml::table &tbl) {
// ----------------------------- [merge] ----------------------------
    try {
        const auto *merge = tbl["merge"].as_table();
        if (!merge) {
            throw std::runtime_error("missing [merge] table");
        }
        cfg_.max_boxes_in_cluster = read_required<int>(*merge, "max_boxes_in_cluster");
        cfg_.neighbor_iou_th = read_required<float>(*merge, "neighbor_iou_th");
        cfg_.center_dist_factor = read_required<float>(*merge, "center_dist_factor");
        cfg_.max_area_multiplier = read_required<float>(*merge, "max_area_multiplier");
        return true;

    } catch (const std::exception &e) {
        std::cerr << "merge config load failed  " << e.what() << std::endl;
        return false;
    }
};

