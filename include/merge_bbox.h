#pragma  once

#include "config.h"


class MergeBbox {
public:
    MergeBbox(toml::table &tbl) {
        load_merge_config(tbl);
    }

    struct MergeConfig {
        // Максимум bbox в кластере
        int max_boxes_in_cluster = 2;

        // IoU соседства
        float neighbor_iou_th = 0.05f;

        // Коэффициент расстояния между центрами
        float center_dist_factor = 5.5f;

        // Максимальный рост площади merged bbox
        float max_area_multiplier = 3.0f;
    };

    MergeConfig cfg_;


    std::vector<cv::Rect2f> merge_detections(const std::vector<cv::Rect2f> &dets);

private:
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

    bool load_merge_config(const toml::table &tbl);
};