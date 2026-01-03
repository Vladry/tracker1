#include "automatic_motion_detector.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {
    constexpr int kHistorySize = 3;
    constexpr int kDiffThreshold = 25;
    constexpr double kMinArea = 60.0;
}

AutomaticMotionDetector::AutomaticMotionDetector(const ManualMotionDetector* detector)
        : detector_(detector) {}

void AutomaticMotionDetector::set_detector(const ManualMotionDetector* detector) {
    detector_ = detector;
}

void AutomaticMotionDetector::set_tracked_boxes(const std::vector<cv::Rect2f>& tracked_boxes) {
    tracked_boxes_ = tracked_boxes;
}

cv::Mat AutomaticMotionDetector::to_gray(const cv::Mat& frame) {
    if (frame.channels() == 3) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }
    return frame.clone();
}

cv::Point2f AutomaticMotionDetector::rect_center(const cv::Rect& rect) {
    return cv::Point2f(rect.x + rect.width * 0.5f, rect.y + rect.height * 0.5f);
}

std::vector<cv::Point2f> AutomaticMotionDetector::detect_by_motion(const cv::Mat& frame) {
    std::vector<cv::Point2f> points;
    if (frame.empty()) {
        return points;
    }

    gray_history_.push_back(to_gray(frame));
    if (static_cast<int>(gray_history_.size()) > kHistorySize) {
        gray_history_.erase(gray_history_.begin());
    }
    if (gray_history_.size() < 2) {
        return points;
    }

    const cv::Mat& first = gray_history_.front();
    const cv::Mat& last = gray_history_.back();
    cv::Mat diff;
    cv::absdiff(first, last, diff);
    cv::threshold(diff, diff, kDiffThreshold, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(diff, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < kMinArea) {
            continue;
        }
        cv::Rect rect = cv::boundingRect(contour);
        if (rect.area() <= 0) {
            continue;
        }
        points.push_back(rect_center(rect));
    }

    return points;
}

bool AutomaticMotionDetector::find_nearest(const cv::Point2f& reference,
                                           const std::vector<cv::Point2f>& points,
                                           cv::Point2f& out_point) const {
    float best_dist = std::numeric_limits<float>::max();
    bool found = false;

    for (const auto& point : points) {
        bool tracked = false;
        for (const auto& rect : tracked_boxes_) {
            if (rect.contains(point)) {
                tracked = true;
                break;
            }
        }
        if (tracked) {
            continue;
        }

        const float dx = point.x - reference.x;
        const float dy = point.y - reference.y;
        const float dist = std::sqrt(dx * dx + dy * dy);
        if (dist < best_dist) {
            best_dist = dist;
            out_point = point;
            found = true;
        }
    }

    return found;
}

bool AutomaticMotionDetector::find_best_candidate(const cv::Mat& frame, int cx, int cy, cv::Point2f& out_point) {
    const cv::Point2f reference(static_cast<float>(cx), static_cast<float>(cy));
    std::vector<cv::Point2f> points = detect_by_motion(frame);
    return find_nearest(reference, points, out_point);
}
