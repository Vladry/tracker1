#pragma once

#include <opencv2/opencv.hpp>
#include <mutex>
#include <string>
#include <vector>
#include "config.h"

struct StaticTarget {
    int id = -1;
    cv::Rect2f bbox;
    float contrast = 0.0f;
    long long created_ms = 0;
};

class StaticTargetManager {
public:
    struct Config {
        int max_targets = 5;
        int click_padding = 6;
        int remove_padding = 6;
        int fallback_box_size = 40;
        float max_area_ratio = 0.1f;
        int click_search_radius = 40;
        bool click_equalize = true;
        int floodfill_lo_diff = 20;
        int floodfill_hi_diff = 20;
        int overlay_ttl_seconds = 3;
        int min_area = 60;
        int min_width = 6;
        int min_height = 6;
        float min_contrast = 5.0f;
    };

    explicit StaticTargetManager(const toml::table& tbl);

    bool handle_right_click(int x, int y, const cv::Mat& frame, long long now_ms);
    void update(cv::Mat& frame, long long now_ms);
    const std::vector<StaticTarget>& targets() const { return targets_; }

private:
    Config cfg_;
    LoggingConfig log_cfg_;
    std::vector<StaticTarget> targets_;
    int next_id_ = 1;
    std::mutex mutex_;

    bool load_config(const toml::table& tbl);
    cv::Rect2f build_roi_from_click(const cv::Mat& frame, int x, int y, cv::Mat* mask) const;
    cv::Rect2f clip_rect(const cv::Rect2f& rect, const cv::Size& size) const;
    float compute_contrast(const cv::Mat& frame, const cv::Rect2f& roi) const;
    bool point_in_rect_with_padding(const cv::Rect2f& rect, int x, int y, int pad) const;

    cv::Mat flood_fill_overlay_;
    cv::Mat flood_fill_mask_;
    long long overlay_expire_ms_ = 0;
};
