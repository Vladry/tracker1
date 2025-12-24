#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "target.h"

// IoU-based tracker WITHOUT Kalman filter.
// - Tracks are updated purely by assignment to detections (IoU >= threshold).
// - If not matched, track is kept for max_missed_frames and then removed.
class TrackerManager {
public:
    // ============================================================================
// Tracker configuration
// ============================================================================
    struct TrackerConfig {
        // IoU для сопоставления
        float iou_threshold = 0.25f;
        // Максимальное число пропущенных кадров без детекции
        int max_missed_frames = 30;
        // Максимумальное кол-во отслеживаемых активных целей
        int max_targets = 10;
    };


    explicit TrackerManager(const TrackerConfig& cfg);

    void reset();

    // Update with detections; returns current targets
    std::vector<Target> update(const std::vector<cv::Rect2f>& detections);

    // Choose target by click point; returns selected ID if hit, else -1
    int pickTargetId(int x, int y) const;

    // Presence check (used to drop selection when track disappears)
    bool hasTargetId(int id) const;

    const std::vector<Target>& targets() const { return targets_; }

private:
    struct Track {
        int id = -1;
        cv::Rect2f bbox;
        int age = 0;
        int missed = 0;
    };

    static float iou(const cv::Rect2f& a, const cv::Rect2f& b);

private:
    TrackerConfig cfg_;
    int next_id_ = 1;
    std::vector<Track> tracks_;
    std::vector<Target> targets_;
};
