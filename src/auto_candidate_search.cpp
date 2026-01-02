#include "auto_candidate_search.h"
#include <cmath>

namespace {
    static inline cv::Mat to_gray(const cv::Mat& frame) {
        if (frame.channels() == 3) {
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            return gray;
        }
        return frame.clone();
    }
}

AutoCandidateSearch::AutoCandidateSearch(const ManualMotionDetector* detector, int timeout_ms)
        : detector_(detector), timeout_ms_(timeout_ms) {}

void AutoCandidateSearch::configure(const ManualMotionDetector* detector, int timeout_ms) {
    detector_ = detector;
    timeout_ms_ = timeout_ms;
}

void AutoCandidateSearch::reset() {
    started_ = false;
    active_ = false;
    start_ms_ = 0;
    roi_ = {};
    gray_frames_.clear();
}

void AutoCandidateSearch::start(const cv::Point2f& last_pos, long long now_ms, const cv::Mat& frame) {
    if (!started_) {
        started_ = true;
        start_ms_ = now_ms;
        last_pos_ = last_pos;
    }

    if (active_ || !detector_ || frame.empty()) {
        return;
    }

    const int cx = static_cast<int>(std::round(last_pos_.x));
    const int cy = static_cast<int>(std::round(last_pos_.y));
    roi_ = detector_->make_click_roi(frame, cx, cy);
    if (roi_.area() <= 0) {
        return;
    }

    gray_frames_.clear();
    gray_frames_.push_back(to_gray(frame));
    active_ = true;
}

bool AutoCandidateSearch::update(const cv::Mat& frame, cv::Rect2f& out_bbox) {
    if (!active_ || !detector_ || frame.empty()) {
        return false;
    }

    gray_frames_.push_back(to_gray(frame));
    const int required = detector_->required_frames();
    if (static_cast<int>(gray_frames_.size()) > required) {
        gray_frames_.erase(gray_frames_.begin());
    }
    if (static_cast<int>(gray_frames_.size()) < required) {
        return false;
    }

    if (detector_->build_candidate(gray_frames_, roi_, frame.size(), out_bbox, nullptr, nullptr)) {
        active_ = false;
        gray_frames_.clear();
        return true;
    }

    return false;
}

bool AutoCandidateSearch::timed_out(long long now_ms) const {
    if (!started_) {
        return false;
    }
    return (now_ms - start_ms_) > static_cast<long long>(timeout_ms_);
}
