#include <iostream>
#include <opencv2/core/types.hpp>
#include "merge_bbox.h"



std::vector<cv::Rect2f> MergeBbox::merge_detections(const std::vector<cv::Rect2f> &dets) {
    std::vector<cv::Rect2f> out;
    if (dets.empty())
        return out;

    std::vector<int> idx(dets.size());
    for (size_t i = 0; i < dets.size(); ++i)
        idx[i] = (int) i;

    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) {
                  return dets[a].area() > dets[b].area();
              });

    std::vector<char> used(dets.size(), 0);

    for (int seed: idx) {
        if (used[seed])
            continue;

        cv::Rect2f merged = dets[seed];
        used[seed] = 1;

        bool grew = true;
        int merged_count = 1;

        while (grew && merged_count < cfg_.max_boxes_in_cluster) {
            grew = false;
            int best_j = -1;
            float best_score = 0.f;

            for (size_t j = 0; j < dets.size(); ++j) {
                if (used[j])
                    continue;

                float v_iou = iou_rect(merged, dets[j]);
                float d = center_dist(merged, dets[j]);
                float s = ref_size(merged);

                bool neighbor =
                        (v_iou >= cfg_.neighbor_iou_th) ||
                        (d <= cfg_.center_dist_factor * s);

                if (!neighbor)
                    continue;

                float score = v_iou + 0.001f / (1.0f + d);
                if (score > best_score) {
                    best_score = score;
                    best_j = (int) j;
                }
            }

            if (best_j >= 0) {
                used[best_j] = 1;
                merged |= dets[best_j];
                merged_count++;
                grew = true;
            }
        }

        out.push_back(merged);
    }

    return out;
}


// ===================== GEOMETRY HELPERS =====================

static inline float iou_rect(const cv::Rect2f &a, const cv::Rect2f &b) {
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    return (uni > 0.f) ? (inter / uni) : 0.f;
}

static inline float center_dist(const cv::Rect2f &a, const cv::Rect2f &b) {
    float ax = a.x + a.width * 0.5f;
    float ay = a.y + a.height * 0.5f;
    float bx = b.x + b.width * 0.5f;
    float by = b.y + b.height * 0.5f;
    float dx = ax - bx;
    float dy = ay - by;
    return std::sqrt(dx * dx + dy * dy);
}

static inline float ref_size(const cv::Rect2f &r) {
    return std::max(10.0f, 0.5f * (r.width + r.height));
}

bool MergeBbox::load_merge_config(const toml::table &tbl) {
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
}
