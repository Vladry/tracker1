#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <mutex>
#include <vector>
#include "config.h"
#include "target.h"

class ManualTrackerManager {
public:
    struct Config {
        int max_targets = 5;
        int click_padding = 6;
        int remove_padding = 6;
        int fallback_box_size = 40;
        float max_area_ratio = 0.1f;
        int motion_search_radius = 30;
        int motion_diff_threshold = 25;
        int click_search_radius = 80;
        int floodfill_lo_diff = 20;
        int floodfill_hi_diff = 20;
        int min_area = 200;
        int min_width = 10;
        int min_height = 10;
        int search_radius = 60;
        float match_threshold = 0.7f;
        bool update_template = true;
        int max_lost_ms = 1500;
        bool auto_reacquire_nearest = true;
        int reacquire_delay_ms = 2000;
        int reacquire_max_distance_px = 120;
        float kalman_process_noise = 1e-2f;
        float kalman_measurement_noise = 1e-1f;
    };

    explicit ManualTrackerManager(const toml::table& tbl);

    void update(const cv::Mat& frame, long long now_ms);
    bool handle_click(int x, int y, const cv::Mat& frame, long long now_ms);
    const std::vector<Target>& targets() const { return targets_; }

private:
    struct ManualTrack {
        int id = -1;
        cv::Rect2f bbox;
        cv::Ptr<cv::Tracker> tracker;
        cv::Mat template_gray;
        int age_frames = 0;
        int missed_frames = 0;
        long long last_seen_ms = 0;
        bool predicted = false;
        cv::KalmanFilter kf;
        bool kf_ready = false;
    };

    Config cfg_;
    std::vector<ManualTrack> tracks_;
    std::vector<Target> targets_;
    int next_id_ = 1;
    std::mutex mutex_;
    cv::Mat prev_gray_;

    bool load_config(const toml::table& tbl);
    cv::Rect2f build_roi_from_click(const cv::Mat& frame, int x, int y) const;
    cv::Rect2f find_motion_roi(const cv::Mat& frame, int x, int y);
    cv::Rect2f clip_rect(const cv::Rect2f& rect, const cv::Size& size) const;
    void init_kalman(ManualTrack& track, const cv::Point2f& center);
    void predict_kalman(ManualTrack& track);
    void correct_kalman(ManualTrack& track, const cv::Point2f& center);
    bool try_reacquire_with_template(ManualTrack& track, const cv::Mat& frame);
    bool point_in_rect_with_padding(const cv::Rect2f& rect, int x, int y, int pad) const;
    void refresh_targets();
};
