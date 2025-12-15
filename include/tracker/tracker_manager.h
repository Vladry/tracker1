#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>

#include "target.h"

class TrackerManager {
public:
    struct Config {
        // ---- limits / lifecycle ----
        int   max_targets = 10;

        // delete if not seen for this long (seconds)
        float occlusion_timeout_sec = 2.0f;

        // confirmation (optional)
        int   confirm_hits = 2;

        // ---- detector seeding gate (main filters too, but keep here for safety) ----
        float seed_overlap_iou = 0.30f; // if new detection overlaps existing track above this, ignore it

        // ---- Template tracking (core of "tracker follows target") ----
        int   tmpl_patch_w = 32;
        int   tmpl_patch_h = 32;

        // search window expansion around current bbox (pixels)
        int   tmpl_search_px = 40;

        // matchTemplate threshold (TM_CCOEFF_NORMED), 0..1, higher is stricter
        float tmpl_min_score = 0.60f;

        // template adaptation (0..1), small value recommended
        float tmpl_update_alpha = 0.05f;

        // ignore too small targets for template tracking
        int   min_template_area_px = 200;

        // ---- debug ----
        bool  enable_template_tracking = true;
    };

    explicit TrackerManager(const Config& cfg);

    void reset();

    // Detector provides ONLY "new targets" here. Tracker follows existing on every frame.
    std::vector<Target> update(const cv::Mat& frame_bgr,
                               const std::vector<cv::Rect2f>& new_detections);

    const std::vector<Target>& targets() const { return targets_; }

    int pickTargetId(int x, int y) const;

private:
    struct Track {
        int id = -1;

        cv::Rect2f bbox;

        int hits = 0;
        bool confirmed = false;

        // template tracking state
        cv::Mat tmpl_gray; // CV_8UC1, size = tmpl_patch_h x tmpl_patch_w

        // occlusion timer
        std::chrono::steady_clock::time_point last_seen;
    };

    Config cfg_;
    int next_id_ = 1;

    std::vector<Track> tracks_;
    std::vector<Target> targets_;

private:
    static float iou(const cv::Rect2f& a, const cv::Rect2f& b);

    bool extractTemplate(const cv::Mat& frame_bgr, const cv::Rect2f& bbox, cv::Mat& out_tmpl) const;
    bool templateTrackOne(const cv::Mat& frame_bgr, Track& tr) const;
    void updateTemplate(Track& tr, const cv::Mat& new_tmpl) const;

    bool overlapsExisting(const cv::Rect2f& det) const;

    void rebuildTargets();
};
