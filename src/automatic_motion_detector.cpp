#include "automatic_motion_detector.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {
    constexpr int kHistorySize = 5;
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

void AutomaticMotionDetector::set_detection_params(int iterations,
                                                   float diffusion_pixels,
                                                   float cluster_ratio_threshold) {
    detection_iterations_ = std::max(1, iterations);
    diffusion_pixels_ = std::max(1.0f, diffusion_pixels);
    cluster_ratio_threshold_ = std::max(0.0f, std::min(cluster_ratio_threshold, 1.0f));
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
    std::vector<cv::Point2f> all_points;
    all_points.reserve(static_cast<size_t>(detection_iterations_) * 4);
    for (int i = 0; i < detection_iterations_; ++i) {
        std::vector<cv::Point2f> points = detect_by_motion(frame);
        all_points.insert(all_points.end(), points.begin(), points.end());
    }
    if (all_points.empty()) {
        return false;
    }

    const float radius_sq = diffusion_pixels_ * diffusion_pixels_;
    std::vector<bool> clustered(all_points.size(), false);
    std::vector<cv::Point2f> filtered_points;
    const int total_points = static_cast<int>(all_points.size());
    for (size_t i = 0; i < all_points.size(); ++i) {
        if (clustered[i]) {
            continue;
        }
        cv::Point2f sum(0.0f, 0.0f);
        int count = 0;
        for (size_t j = i; j < all_points.size(); ++j) {
            const float dx = all_points[j].x - all_points[i].x;
            const float dy = all_points[j].y - all_points[i].y;
            if (dx * dx + dy * dy <= radius_sq) {
                clustered[j] = true;
                sum += all_points[j];
                count += 1;
            }
        }
        if (count > 0) {
            const float ratio = static_cast<float>(count) / static_cast<float>(total_points);
            if (ratio >= cluster_ratio_threshold_) {
                filtered_points.push_back(cv::Point2f(sum.x / count, sum.y / count));
            }
        }
    }

    if (filtered_points.empty()) {
        return false;
    }

    return find_nearest(reference, filtered_points, out_point);
}
