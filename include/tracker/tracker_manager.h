#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "tracker/target.h"

// TrackerManager is the single entry-point for tracking state.
// It is intentionally lightweight for integration with your current main/overlay flow:
//   - update(detections) is called once per frame
//   - targets() returns current list for overlay
//   - pickTargetId(x,y) supports mouse selection
//
// Internally it uses a per-target Kalman Filter for center position (x,y,vx,vy).
class TrackerManager {
public:
    struct Config {
        // Association / matching
        float match_iou_threshold = 0.25f;   // primary "good match" criterion
        float assoc_iou_min       = 0.05f;   // allow weaker IoU if distance gate passes
        float assoc_dist_max_px   = 80.0f;   // center distance gate
        float score_iou_w         = 0.65f;   // weighted score for greedy matching
        float score_dist_w        = 0.35f;

        // Track lifecycle
        int confirm_hits              = 3;   // hits needed to mark track as confirmed
        int max_missed_frames         = 30;  // base deletion threshold
        int stationary_grace_frames   = 60;  // extra keep time when target stops moving
        float stationary_speed_thresh = 0.6f; // px/frame; below => treated as stationary (KF velocity)

        // Capacity
        int max_targets = 10;

        // Kalman tuning
        float process_noise = 1e-2f;
        float meas_noise    = 1e-1f;
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
        int id = -1;
        cv::KalmanFilter kf;
        cv::Rect2f bbox;

        int age = 0;
        int missed = 0;

        int hits = 0;
        bool confirmed = false;
    };

    static cv::KalmanFilter makeKF(float px, float py, float vx, float vy,
                                   float process_noise, float meas_noise);

    static float trackSpeedPxPerFrame(const TrackKF& t);

    // Greedy matching: returns for each detection index -> matched track index (or -1)
    void associateGreedy(const std::vector<cv::Rect2f>& detections,
                         std::vector<int>& det_to_track);

    void spawnNewTracks(const std::vector<cv::Rect2f>& detections,
                        const std::vector<int>& det_to_track);

    bool shouldDelete(const TrackKF& t) const;

private:
    Config cfg_;
    int next_id_ = 1;
    std::vector<TrackKF> tracks_;
    std::vector<Target> targets_;
};
