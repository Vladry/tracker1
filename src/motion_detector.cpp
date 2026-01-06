#include "motion_detector.h"

#include <algorithm>
#include <cmath>
#include <limits>

void MotionDetector::set_detection_params(int iterations,
                                          float diffusion_pixels,
                                          float cluster_ratio_threshold) {
    detection_iterations_ = std::max(1, iterations);
    diffusion_pixels_ = std::max(1.0f, diffusion_pixels);
    cluster_ratio_threshold_ = std::max(0.0f, std::min(cluster_ratio_threshold, 1.0f));
    if (detection_history_.size() > static_cast<size_t>(detection_iterations_)) {
        while (detection_history_.size() > static_cast<size_t>(detection_iterations_)) {
            detection_history_.pop_front();
        }
    }
}

void MotionDetector::set_motion_params(int history_size, int diff_threshold, double min_area) {
    history_size_ = std::max(2, history_size);
    diff_threshold_ = std::max(0, diff_threshold);
    min_area_ = std::max(0.0, min_area);
    if (gray_history_.size() > static_cast<size_t>(history_size_)) {
        while (gray_history_.size() > static_cast<size_t>(history_size_)) {
            gray_history_.erase(gray_history_.begin());
        }
    }
}

void MotionDetector::set_update_period_ms(int period_ms) {
    update_period_ms_ = std::max(1, period_ms);
}

void MotionDetector::set_binarize_max_value(int max_value) {
    binarize_max_value_ = std::max(1, max_value);
}

void MotionDetector::reset() {
    gray_history_.clear();
    detection_history_.clear();
    filtered_points_.clear();
    last_update_ms_ = 0;
}

// Цепочка 2 (фоновая автодетекция):
// 1) каждый кадр добавляется в историю gray_history_,
// 2) с периодом update_period_ms_ выполняется detect_by_motion(),
// 3) результаты агрегируются и кластеризуются в rebuild_filtered_points(),
// 4) detections() возвращает итоговый пул кандидатов.
void MotionDetector::update(const cv::Mat& frame, long long now_ms) {
    if (frame.empty()) {
        return;
    }

    gray_history_.push_back(to_gray(frame));
    if (static_cast<int>(gray_history_.size()) > history_size_) {
        gray_history_.erase(gray_history_.begin());
    }
    if (gray_history_.size() < 2) {
        return;
    }

    if (last_update_ms_ > 0 && now_ms - last_update_ms_ < update_period_ms_) {
        return;
    }
    last_update_ms_ = now_ms;

    std::vector<cv::Point2f> points = detect_by_motion(frame);
    detection_history_.push_back(std::move(points));
    while (detection_history_.size() > static_cast<size_t>(detection_iterations_)) {
        detection_history_.pop_front();
    }
    rebuild_filtered_points();
}

cv::Mat MotionDetector::to_gray(const cv::Mat& frame) {
    if (frame.channels() == 3) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }
    return frame.clone();
}

cv::Point2f MotionDetector::rect_center(const cv::Rect& rect) {
    return cv::Point2f(rect.x + rect.width * 0.5f, rect.y + rect.height * 0.5f);
}

// Находит центры движущихся областей по разности первого и последнего кадров истории.
std::vector<cv::Point2f> MotionDetector::detect_by_motion(const cv::Mat& frame) {
    std::vector<cv::Point2f> points;
    if (frame.empty()) {
        return points;
    }

    if (gray_history_.size() < 2) {
        return points;
    }

    const cv::Mat& first = gray_history_.front();
    const cv::Mat& last = gray_history_.back();
    cv::Mat diff;
    cv::absdiff(first, last, diff);
    cv::threshold(diff, diff, diff_threshold_, binarize_max_value_, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(diff, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < min_area_) {
            continue;
        }
        cv::Rect rect = cv::boundingRect(contour);
        if (rect.area() <= 0) {
            continue;
        }
        points.push_back(rect_center(rect));
    }

//    std::cout<<"batch of detected points: "<<points<<std::endl;

    return points;
}

void MotionDetector::rebuild_filtered_points() {
    std::vector<cv::Point2f> all_points;
    for (const auto& sample : detection_history_) {
        all_points.insert(all_points.end(), sample.begin(), sample.end());
    }
    filtered_points_.clear();
    if (all_points.empty()) {
        return;
    }

    const float radius_sq = diffusion_pixels_ * diffusion_pixels_;
    std::vector<bool> clustered(all_points.size(), false);
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
                filtered_points_.push_back(cv::Point2f(sum.x / count, sum.y / count));
            }
        }
    }
}
