#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "manual_motion_detector.h"

class AutoCandidateSearch {
public:
    AutoCandidateSearch() = default;
    AutoCandidateSearch(const ManualMotionDetector* detector, int timeout_ms);

    void configure(const ManualMotionDetector* detector, int timeout_ms);
    void reset();

    void start(const cv::Point2f& last_pos, long long now_ms, const cv::Mat& frame);
    bool update(const cv::Mat& frame, cv::Rect2f& out_bbox);

    bool active() const { return active_; }
    bool timed_out(long long now_ms) const;

private:
    const ManualMotionDetector* detector_ = nullptr;
    int timeout_ms_ = 0;
    bool started_ = false;
    bool active_ = false;
    long long start_ms_ = 0;
    cv::Rect roi_;
    std::vector<cv::Mat> gray_frames_;
    cv::Point2f last_pos_{0.0f, 0.0f};
};
