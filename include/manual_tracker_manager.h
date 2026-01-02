#pragma once

#include <array>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <mutex>
#include <string>
#include <vector>
#include "auto_candidate_search.h"
#include "config.h"
#include "manual_motion_detector.h"
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
        int click_capture_size = 80;
        int motion_frames = 3;
        int overlay_ttl_seconds = 3;
        int tracker_init_padding = 10;
        int tracker_min_size = 24;
        float motion_min_magnitude = 0.4f;
        float motion_angle_tolerance_deg = 20.0f;
        float motion_mag_tolerance_px = 3.0f;
        float tracker_motion_min_ratio = 0.02f;
        int tracker_motion_grace_frames = 3;
        int lost_bbox_ttl_ms = 3000;
        int reacquire_fallback_max_distance_px = 300;
        int reacquire_kalman_radius_px = 120;
        int reacquire_near_radius_px = 200;
        bool use_kalman = false;
        bool click_equalize = true;
        bool floodfill_fill_overlay = true;
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
        int candidate_search_timeout_ms = 1000;
        std::string tracker_type = "KCF";
        float kalman_process_noise = 1e-2f;
        float kalman_measurement_noise = 1e-1f;
    };

    explicit ManualTrackerManager(const toml::table& tbl);

    void update(cv::Mat& frame, long long now_ms);
    bool handle_click(int x, int y, const cv::Mat& frame, long long now_ms);
    const std::vector<Target>& targets() const { return targets_; }

private:
    struct ManualTrack {
        int id = -1;
        cv::Rect2f bbox;
        cv::Ptr<cv::Tracker> tracker;
        cv::Mat template_gray;
        bool is_dynamic = false;
        int age_frames = 0;
        int missed_frames = 0;
        long long last_seen_ms = 0;
        bool logged_reacquire_ready = false;
        bool predicted = false;
        cv::KalmanFilter kf;
        bool kf_ready = false;
        cv::Point2f predicted_center;
        bool predicted_center_ready = false;
        int stale_frames = 0;
        long long lost_since_ms = 0;
        cv::Rect reacquire_roi;
        std::vector<cv::Mat> reacquire_gray_frames;
        int reacquire_stage = 0;
        std::array<bool, 3> visibility_history{true, true, true};
        size_t visibility_index = 0;
        cv::Point2f last_known_center{0.0f, 0.0f};
        AutoCandidateSearch candidate_search;
    };

    struct PendingClick {
        cv::Point2i click;
        cv::Rect roi;
        long long start_ms = 0;
        std::vector<cv::Mat> gray_frames;
    };

    Config cfg_;
    LoggingConfig log_cfg_;
    std::vector<ManualTrack> tracks_;
    std::vector<Target> targets_;
    std::vector<PendingClick> pending_clicks_;
    int next_id_ = 1;
    std::mutex mutex_;
    cv::Mat prev_gray_;
    cv::Mat flood_fill_overlay_;
    cv::Mat flood_fill_mask_;
    long long overlay_expire_ms_ = 0;
    ManualMotionDetector motion_detector_;

    bool load_config(const toml::table& tbl);
    cv::Rect2f clip_rect(const cv::Rect2f& rect, const cv::Size& size) const;
    cv::Ptr<cv::Tracker> create_tracker() const;
    bool point_in_rect_with_padding(const cv::Rect2f& rect, int x, int y, int pad) const;
    void record_visibility(ManualTrack& track, bool visible);
    bool has_recent_visibility_loss(const ManualTrack& track) const;
    void refresh_targets();
};
