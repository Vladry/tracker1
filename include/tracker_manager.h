#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "target.h"

class TrackerManager {
public:
    struct Config {
        float iou_threshold = 0.25f;
        int max_missed_frames = 30; // hold track for this many frames without detection
        int max_targets = 10;
        float process_noise = 1e-2f;
        float meas_noise = 1e-1f;
    };

    explicit TrackerManager(const Config& cfg);

    void reset();

    // Update with detections; returns current targets
    std::vector<Target> update(const std::vector<cv::Rect2f>& detections);

    // Choose target by click point; returns selected ID if hit, else -1
    int pickTargetId(int x, int y) const;

    const std::vector<Target>& targets() const { return targets_; }

private:
    struct TrackKF {
        int id;
        cv::KalmanFilter kf;
        cv::Rect2f bbox;
        int age = 0;
        int missed = 0;
    };

    static float iou(const cv::Rect2f& a, const cv::Rect2f& b);

private:
    Config cfg_;
    int next_id_ = 1;
    std::vector<TrackKF> tracks_;
    std::vector<Target> targets_;
};
