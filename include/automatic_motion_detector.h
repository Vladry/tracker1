#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "auto_detection_provider.h"
#include "manual_motion_detector.h"

class AutomaticMotionDetector {
public:
    explicit AutomaticMotionDetector(const ManualMotionDetector* detector = nullptr);

    void set_detector(const ManualMotionDetector* detector);
    void set_detection_provider(AutoDetectionProvider* provider);
    void set_tracked_boxes(const std::vector<cv::Rect2f>& tracked_boxes);
    void set_reserved_detection_points(const std::vector<cv::Point2f>& reserved_points);
    void set_detection_params(int iterations, float diffusion_pixels, float cluster_ratio_threshold);
    void set_motion_params(int history_size, int diff_threshold, double min_area);
    void reset_state();

    bool find_best_candidate(int cx, int cy, cv::Point2f& out_point) const;
    bool find_nearest(const cv::Point2f& reference,
                      const std::vector<cv::Point2f>& points,
                      cv::Point2f& out_point) const;

private:
    const ManualMotionDetector* detector_ = nullptr;
    AutoDetectionProvider* detection_provider_ = nullptr;
    std::vector<cv::Rect2f> tracked_boxes_;
    std::vector<cv::Point2f> reserved_detection_points_;
    int detection_iterations_ = 10;
    float diffusion_pixels_ = 100.0f;
    float cluster_ratio_threshold_ = 0.9f;
    int history_size_ = 5;
    int diff_threshold_ = 25;
    double min_area_ = 60.0;
    float reserved_detection_radius_ = 200.0f;
};
