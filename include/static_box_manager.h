#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <deque>
#include <unordered_map>
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
    int missed_frames = 0;

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
    // количество одновременных статических боксов
    static constexpr int static_boxes_max_amount = 1;

    static constexpr size_t kTrajectoryHistorySize = 8;
    static constexpr float kDirectionSimilarityThreshold = 0.5f;

    struct StaticBoxConfig {
// ============================================================================
// Static rebind configuration
// ============================================================================
        // Автоматическая перепривязка static bbox
        bool auto_rebind = true;

        // Таймаут ожидания новой цели (мс)
        int rebind_timeout_ms = 1200;

        // IoU-порог родительской привязки
        float parent_iou_th = 0.15f;

        // Порог уверенности перепривязки
        float reattach_score_th = 0.20f;

        // Максимально допустимое количество пропущенных кадров
        int max_missed_frames = 3;
    };

    struct TrajectoryHistory {
        std::deque<cv::Point2f> points;
        int missed_frames = 0;
    };

    StaticBoxConfig cfg_;
    std::vector <static_box> boxes_;
    int next_id_ = 1;
    std::unordered_map<int, TrajectoryHistory> trajectories_;

    void update_trajectories(
            const std::vector<cv::Rect2f> &boxes,
            const std::vector<int> &ids
    );

    int find_nearest_with_direction(
            const static_box &sb,
            const std::vector <cv::Rect2f> &boxes,
            const std::vector<int> &ids,
            const cv::Point2f &reference_dir,
            bool has_reference_dir
    ) const;

    bool load_static_rebind_config(const toml::table &tbl);
};
