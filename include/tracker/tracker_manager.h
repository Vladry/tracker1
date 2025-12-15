#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#include "target.h"

// TrackerManager:
// - получает детекции bbox (после merging blobs)
// - ведёт треки с Kalman
// - удерживает цели при остановке движения через presence confirmation в ROI
class TrackerManager {
public:
    struct Config {
        // Association / lifecycle
        float iou_threshold        = 0.25f; // minimum IOU to accept detection match
        int   max_missed_frames    = 30;    // hard delete when missed too long (unconfirmed)
        int   max_targets          = 10;    // global limit

        // Kalman noise
        float process_noise        = 1e-2f;
        float meas_noise           = 1e-1f;

        // Confirmation & stationary holding
        int   confirm_hits         = 3;     // hits before track becomes confirmed
        int   stationary_grace_frames = 60; // extra frames to keep confirmed tracks without detections

        // Presence confirmation (appearance)
        int   appearance_patch_w   = 24;    // ROI resized to WxH
        int   appearance_patch_h   = 24;
        float appearance_l1_thresh = 18.0f; // mean L1 per pixel threshold (tune in main)
        float appearance_update_alpha = 0.10f; // how fast reference adapts when matched (0..1)
        int   min_presence_area_px = 120;   // ignore too small bbox for presence check
    };

    explicit TrackerManager(const Config& cfg);

    void reset();

    // IMPORTANT: now requires current frame to do presence confirmation.
    // detections are in full-frame coordinates (cv::Rect2f).
    std::vector<Target> update(const cv::Mat& frame_bgr,
                               const std::vector<cv::Rect2f>& detections);

    const std::vector<Target>& targets() const { return targets_; }

    int pickTargetId(int x, int y) const;

private:
    struct TrackKF {
        int id = -1;

        cv::KalmanFilter kf;
        cv::Rect2f bbox;

        int age = 0;       // frames since birth
        int missed = 0;    // consecutive misses
        int hits = 0;      // total successful matches
        bool confirmed = false;

        // Appearance reference (grayscale patch), used when detector outputs nothing
        cv::Mat appearance_ref;     // CV_8UC1, size = (appearance_patch_h x appearance_patch_w)
        bool appearance_valid = false;
    };

    Config cfg_;
    int next_id_ = 1;

    std::vector<TrackKF> tracks_;
    std::vector<Target> targets_;

    static float iou(const cv::Rect2f& a, const cv::Rect2f& b);

    // presence confirmation helpers
    bool extractAppearanceRef(const cv::Mat& frame_bgr, const cv::Rect2f& bbox, cv::Mat& out_ref) const;
    bool appearanceMatches(const cv::Mat& frame_bgr, const TrackKF& t) const;
    void updateAppearanceRef(const cv::Mat& frame_bgr, TrackKF& t) const;

    static cv::KalmanFilter makeKF(float px, float py, float vx, float vy,
                                   float process_noise, float meas_noise);

    void rebuildTargets();
};
