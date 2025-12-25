#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include "config.h"

enum class static_box_state {
    attached,          // уверенно привязан
    pending_rebind,    // кандидат найден, ожидаем подтверждения (future)
    lost               // цель потеряна
};

struct static_box {
    int id;
    cv::Rect2f rect;

    int last_dynamic_id;
    float confidence;

    static_box_state state;

    std::chrono::steady_clock::time_point last_seen;
};


class StaticBoxManager {
public:
    explicit StaticBoxManager(const toml::table &tbl);

    void on_mouse_click(
            int x, int y,
            const std::vector <cv::Rect2f> &dynamic_boxes,
            const std::vector<int> &dynamic_ids
    );

    void update(
            const std::vector <cv::Rect2f> &dynamic_boxes,
            const std::vector<int> &dynamic_ids
    );

    const std::vector <static_box> &boxes() const { return boxes_; }

    void static_mgr(); //TODO реализовать!


private:
    struct StaticBoxConfig {
        bool auto_rebind_on_loss;
        int rebind_timeout_ms;
        float parent_iou_th;
        float reattach_score_th;
// ============================================================================
// Static rebind configuration
// ============================================================================
        // Автоматическая перепривязка static bbox
        bool auto_rebind = true;

        // Таймаут ожидания новой цели (мс)
        int rebind_timeout_ms = 1200;

        // Вес расстояния
        float distance_weight = 1.0f;

        // Вес площади
        float area_weight = 1.0f;

        // Во сколько раз новая цель должна быть крупнее
        float larger_area_factor = 2.0f;

        // Максимальная допустимая дистанция до крупной цели
        float max_large_target_dist_frac = 0.2f;

        // IoU-порог родительской привязки
        float parent_iou_th = 0.15f;

        // Порог уверенности перепривязки
        float reattach_score_th = 0.20f;
    };


    StaticBoxConfig cfg_;
    std::vector <static_box> boxes_;
    int next_id_ = 1;

    int find_best_match(
            const static_box &sb,
            const std::vector <cv::Rect2f> &boxes,
            const std::vector<int> &ids,
            float &out_score
    ) const;

    int find_nearest(
            const static_box &sb,
            const std::vector <cv::Rect2f> &boxes
    ) const;

    bool load_static_rebind_config(const toml::table &tbl);
};
