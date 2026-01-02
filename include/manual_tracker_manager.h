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
        int motion_diff_threshold = 25;
        int click_capture_size = 80;
        int motion_frames = 3;
        int overlay_ttl_seconds = 3;
        int tracker_init_padding = 10;
        int tracker_min_size = 24;
        float motion_min_magnitude = 0.4f;
        float motion_mag_tolerance_px = 3.0f;
        bool floodfill_fill_overlay = true;
        int floodfill_lo_diff = 20;
        int floodfill_hi_diff = 20;
        int min_area = 200;
        int min_width = 10;
        int min_height = 10;
        std::string tracker_type = "KCF";
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
        int age_frames = 0;
        long long lost_since_ms = 0;
        std::array<bool, 3> visibility_history{true, true, true};
        size_t visibility_index = 0;
        cv::Point2f last_known_center{0.0f, 0.0f};
        AutoCandidateSearch candidate_search;
    };

    struct PendingClick {
        cv::Rect roi;
        std::vector<cv::Mat> gray_frames;
    };

    Config cfg_;
    LoggingConfig log_cfg_;
    std::vector<ManualTrack> tracks_;
    std::vector<Target> targets_;
    std::vector<PendingClick> pending_clicks_;
    int next_id_ = 1;
    std::mutex mutex_;
    cv::Mat flood_fill_overlay_;
    cv::Mat flood_fill_mask_;
    long long overlay_expire_ms_ = 0;
    ManualMotionDetector motion_detector_;

    bool load_config(const toml::table& tbl);
    cv::Ptr<cv::Tracker> create_tracker() const;
    bool point_in_rect_with_padding(const cv::Rect2f& rect, int x, int y, int pad) const;
    void record_visibility(ManualTrack& track, bool visible);
    bool has_recent_visibility_loss(const ManualTrack& track) const;
    void refresh_targets();
};
