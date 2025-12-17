#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>

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

struct static_box_config {
    bool auto_rebind_on_loss;
    int rebind_timeout_ms;

    float parent_iou_th;
    float reattach_score_th;
};

class static_box_manager {
public:
    explicit static_box_manager(const static_box_config& cfg);

    void on_mouse_click(
            int x, int y,
            const std::vector<cv::Rect2f>& dynamic_boxes,
            const std::vector<int>& dynamic_ids
    );

    void update(
            const std::vector<cv::Rect2f>& dynamic_boxes,
            const std::vector<int>& dynamic_ids
    );

    const std::vector<static_box>& boxes() const { return boxes_; }

private:
    static_box_config cfg_;
    std::vector<static_box> boxes_;
    int next_id_ = 1;

    int find_best_match(
            const static_box& sb,
            const std::vector<cv::Rect2f>& boxes,
            const std::vector<int>& ids,
            float& out_score
    ) const;

    int find_nearest(
            const static_box& sb,
            const std::vector<cv::Rect2f>& boxes
    ) const;
};
