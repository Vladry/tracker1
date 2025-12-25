#include <toml++/toml.h>   // ДОЛЖНО БЫТЬ ПЕРВЫМ
#include "config.h"
#include <iostream>
#include <stdexcept>
#include <string_view>


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


