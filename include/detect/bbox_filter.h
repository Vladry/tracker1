#pragma once
#include <opencv2/opencv.hpp>

#include <vector>

namespace detect {

    class BBoxFilter {
    public:
        struct Config {
            // Sanity by area (px)
            float min_area_px = 800.0f;
            float max_area_px = 120000.0f;

            // Max area by fraction of frame
            float max_area_frac = 0.25f;

            // Sanity by size (px)
            float min_w = 12.0f;
            float min_h = 12.0f;
            float max_w = 4000.0f;
            float max_h = 4000.0f;

            // Aspect ratio
            float min_ar = 0.15f;
            float max_ar = 6.0f;

            // Matching
            float match_iou = 0.30f;

            // Area jump gate
            float max_area_jump_ratio = 2.5f;

            // Hysteresis
            int hold_missed_frames = 3;
        };

        explicit BBoxFilter(Config cfg);

        std::vector<cv::Rect2f>
        process(const std::vector<cv::Rect2f>& dets,
                const cv::Size& frame_size);

    private:
        static float iou(const cv::Rect2f& a, const cv::Rect2f& b);

        bool sanityOk(const cv::Rect2f& r, const cv::Size& frame_size) const;
        bool areaJumpOk(const cv::Rect2f& prev, const cv::Rect2f& cur) const;

    private:
        Config cfg_;

        struct TrackedDet {
            cv::Rect2f r;
            int missed = 0;
        };
        std::vector<TrackedDet> prev_;
    };

} // namespace detect
