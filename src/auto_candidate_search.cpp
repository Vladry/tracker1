#include "auto_candidate_search.h"
#include <cmath>

namespace {
    // Переводит кадр в серый, если он цветной.
    // Используется для сравнения последовательности кадров движения.
    static inline cv::Mat to_gray(const cv::Mat& frame) {
        if (frame.channels() == 3) {
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            return gray;
        }
        return frame.clone();
    }
}

AutoCandidateSearch::AutoCandidateSearch(const ClickedTargetShaper* detector,
                                         MotionDetector* motion_detector) {
    configure(detector, motion_detector);
}

// Назначает формирователь цели и фоновый детектор для поиска кандидатов.
void AutoCandidateSearch::configure(const ClickedTargetShaper* detector,
                                    MotionDetector* motion_detector) {
    detector_ = detector;
    motion_detector_ = motion_detector;
    detection_matcher_.set_detector(detector);
    detection_matcher_.set_motion_detector(motion_detector_);
}

void AutoCandidateSearch::configure_motion_filter(int iterations,
                                                  float diffusion_pixels,
                                                  float cluster_ratio_threshold,
                                                  int history_size,
                                                  int diff_threshold,
                                                  double min_area) {
    if (!motion_detector_) {
        return;
    }
    detection_matcher_.set_detection_params(iterations, diffusion_pixels, cluster_ratio_threshold);
    detection_matcher_.set_motion_params(history_size, diff_threshold, min_area);
}

void AutoCandidateSearch::set_reserved_detection_radius(float radius_px) {
    detection_matcher_.set_reserved_detection_radius(radius_px);
}

// Сбрасывает внутреннее состояние поиска кандидатов.
// Останавливает активный сбор кадров и очищает буферы, чтобы начать поиск заново.
void AutoCandidateSearch::reset() {
    started_ = false;
    active_ = false;
    start_ms_ = 0;
    roi_ = {};
    gray_frames_.clear();
    best_candidate_selected_ = false;
    detection_matcher_.reset_state();
}

void AutoCandidateSearch::set_tracked_boxes(const std::vector<cv::Rect2f>& tracked_boxes) {
    tracked_boxes_ = tracked_boxes;
    detection_matcher_.set_tracked_boxes(tracked_boxes_);
}

void AutoCandidateSearch::set_reserved_detection_points(const std::vector<cv::Point2f>& reserved_points) {
    detection_matcher_.set_reserved_detection_points(reserved_points);
}

// Инициализирует поиск вокруг последней позиции цели и сохраняет базовый кадр.
// Шаги:
// 1) запрашивает у DetectionMatcher ближайшую детекцию из MotionDetector,
// 2) строит ROI вокруг найденной точки (или last_pos, если детекций нет),
// 3) начинает накапливать серые кадры для clicked_target_shaper_.
void AutoCandidateSearch::start(const cv::Point2f& last_pos, long long now_ms, const cv::Mat& frame) {
    if (!started_) {
        started_ = true;
        start_ms_ = now_ms;
        last_pos_ = last_pos;
    }

    if (active_ || !detector_ || frame.empty()) {
        return;
    }

    best_candidate_selected_ = false;
    const int cx = static_cast<int>(std::round(last_pos_.x));
    const int cy = static_cast<int>(std::round(last_pos_.y));
    cv::Point2f best_point;
    int roi_x = cx;
    int roi_y = cy;
    if (detection_matcher_.find_best_candidate(cx, cy, best_point)) {
        roi_x = static_cast<int>(std::round(best_point.x));
        roi_y = static_cast<int>(std::round(best_point.y));
        best_candidate_selected_ = true;
    }
    roi_ = detector_->make_click_roi(frame, roi_x, roi_y);
    if (roi_.area() <= 0) {
        return;
    }

    gray_frames_.clear();
    gray_frames_.push_back(to_gray(frame));
    active_ = true;
}

// Добавляет новый кадр и пытается подтвердить движение в ROI.
// Возвращает true и bbox кандидата, когда clicked_target_shaper_ подтверждает цель.
bool AutoCandidateSearch::update(const cv::Mat& frame, cv::Rect2f& out_bbox) {
    if (!active_ || !detector_ || frame.empty()) {
        return false;
    }

    if (!best_candidate_selected_) {
        const int cx = static_cast<int>(std::round(last_pos_.x));
        const int cy = static_cast<int>(std::round(last_pos_.y));
        cv::Point2f best_point;
        if (detection_matcher_.find_best_candidate(cx, cy, best_point)) {
            const int roi_x = static_cast<int>(std::round(best_point.x));
            const int roi_y = static_cast<int>(std::round(best_point.y));
            cv::Rect new_roi = detector_->make_click_roi(frame, roi_x, roi_y);
            if (new_roi.area() > 0) {
                roi_ = new_roi;
                gray_frames_.clear();
                best_candidate_selected_ = true;
            }
        }
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
