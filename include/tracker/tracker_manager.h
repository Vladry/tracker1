#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

#include <vector>
#include <chrono>
#include <string>

#include "target.h"

class TrackerManager {
public:
    struct Config {
        int   max_targets = 10;

        // Visual tracker
        bool  use_csrt = false;          // false = KCF (быстро)
        int   tracker_update_every = 2;  // обновлять tracker раз в N кадров

        // Detection usage (capture + resync)
        float det_match_iou   = 0.20f;
        float spawn_block_iou = 0.25f;

        // Lifecycle
        double occlusion_timeout_sec = 2.0;

        // Resync
        bool  allow_resync = true;
        int   resync_every_n_frames = 15;

        // Debug
        bool  debug_logs = false;
        int   debug_every_n_frames = 60;
    };

    explicit TrackerManager(const Config& cfg);

    void reset();

    std::vector<Target> update(const cv::Mat& frame_bgr,
                               const std::vector<cv::Rect2f>& detections);

    const std::vector<Target>& targets() const { return targets_; }

    int  pickTargetId(int x, int y) const;
    bool hasTargetId(int id) const;

private:
    using Clock = std::chrono::steady_clock;

    struct Track {
        int id = -1;

        cv::Rect2f bbox;
        cv::Ptr<cv::Tracker> tracker;

        bool tracker_ok = false;
        Clock::time_point last_ok;

        int hits = 0;
    };

    Config cfg_;
    int next_id_ = 1;

    std::vector<Track> tracks_;
    std::vector<Target> targets_;

    // frame counter (for throttling visual tracker)
    int frame_index_ = 0;

private:
    static float iou(const cv::Rect2f& a, const cv::Rect2f& b);

    cv::Ptr<cv::Tracker> createTracker() const;
    void initTracker(Track& t, const cv::Mat& frame, const cv::Rect2f& bbox);

    bool spawnBlocked(const cv::Rect2f& det) const;
    void rebuildTargets();
};
