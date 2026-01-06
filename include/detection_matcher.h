#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "motion_detector.h"
#include "clicked_target_shaper.h"

// Выбирает ближайшую детекцию из пула MotionDetector с учётом резерва.
class DetectionMatcher {
public:
    explicit DetectionMatcher(const ClickedTargetShaper* detector = nullptr);

    void set_detector(const ClickedTargetShaper* detector);
    void set_motion_detector(MotionDetector* motion_detector);
    void set_tracked_boxes(const std::vector<cv::Rect2f>& tracked_boxes);
    void set_reserved_detection_points(const std::vector<cv::Point2f>& reserved_points);
    void set_reserved_detection_radius(float radius_px);
    void set_detection_params(int iterations, float diffusion_pixels, float cluster_ratio_threshold);
    void set_motion_params(int history_size, int diff_threshold, double min_area);
    void reset_state();

    bool find_best_candidate(int cx, int cy, cv::Point2f& out_point) const;
    bool find_nearest(const cv::Point2f& reference,
                      const std::vector<cv::Point2f>& points,
                      cv::Point2f& out_point) const;

private:
    const ClickedTargetShaper* detector_ = nullptr; // - формирователь целей по клику (источник ROI).
    MotionDetector* motion_detector_ = nullptr; // - фоновый детектор движения с пулом кандидатов.
    std::vector<cv::Rect2f> tracked_boxes_; // - bbox активных треков, которые исключаются из поиска.
    std::vector<cv::Point2f> reserved_detection_points_; // - центры зарезервированных кандидатов.
    int detection_iterations_ = 10; // - длина истории детекций для кластеризации.
    float diffusion_pixels_ = 100.0f; // - радиус кластеризации детекций (пиксели).
    float cluster_ratio_threshold_ = 0.9f; // - доля точек в кластере для подтверждения кандидата.
    int history_size_ = 5; // - количество кадров в истории diff-детектора движения.
    int diff_threshold_ = 25; // - порог бинаризации diff-кадра для MotionDetector.
    double min_area_ = 60.0; // - минимальная площадь контура для MotionDetector.
    float reserved_detection_radius_ = 200.0f; // - радиус запрета вокруг зарезервированных кандидатов.
};
