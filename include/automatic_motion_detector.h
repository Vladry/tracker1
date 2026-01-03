#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "manual_motion_detector.h"

class AutomaticMotionDetector {
public:
    explicit AutomaticMotionDetector(const ManualMotionDetector* detector = nullptr);

    void set_detector(const ManualMotionDetector* detector);
    void set_tracked_boxes(const std::vector<cv::Rect2f>& tracked_boxes);

    bool find_best_candidate(const cv::Mat& frame, int cx, int cy, cv::Point2f& out_point);
    std::vector<cv::Point2f> detect_by_motion(const cv::Mat& frame);
    bool find_nearest(const cv::Point2f& reference,
                      const std::vector<cv::Point2f>& points,
                      cv::Point2f& out_point) const;

private:
    static cv::Mat to_gray(const cv::Mat& frame);
    static cv::Point2f rect_center(const cv::Rect& rect);

    const ManualMotionDetector* detector_ = nullptr;
    std::vector<cv::Mat> gray_history_;
    std::vector<cv::Rect2f> tracked_boxes_;
};
