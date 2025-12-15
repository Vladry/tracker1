#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <string>

#include "target.h"

class TrackerManager {
public:
    struct Config {
        int   max_targets = 10;

        float occlusion_timeout_sec = 2.0f;
        float stationary_hold_sec   = 30.0f; // держим статичную цель долго
        int   confirm_hits = 2;

        float assoc_iou_threshold = 0.25f;
        float spawn_block_iou     = 0.30f;

        bool  use_kalman = true;
        float kalman_process_noise = 1e-2f;
        float kalman_meas_noise    = 1e-1f;
        float stationary_speed_px = 0.5f;
    };

    explicit TrackerManager(const Config& cfg);

    void reset();

    std::vector<Target> update(
            const cv::Mat& frame,
            const std::vector<cv::Rect2f>& detections
    );

    const std::vector<Target>& targets() const { return targets_; }
    int pickTargetId(int x, int y) const;

private:
    struct Track {
        int id = -1;
        cv::Rect2f bbox;

        int hits = 0;
        bool confirmed = false;

        cv::KalmanFilter kf;
        bool kf_initialized = false;

        std::chrono::steady_clock::time_point last_seen;
    };

    Config cfg_;
    int next_id_ = 1;

    std::vector<Track> tracks_;
    std::vector<Target> targets_;

private:
    static float iou(const cv::Rect2f& a, const cv::Rect2f& b);

    cv::KalmanFilter makeKalman(float cx, float cy) const;
    float speedPx(const Track& t) const;

    void rebuildTargets();
};
