#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "automatic_motion_detector.h"
#include "manual_motion_detector.h"

class AutoCandidateSearch {
public:
    AutoCandidateSearch() = default;
    // Создаёт поисковый модуль с заданным детектором движения.
    AutoCandidateSearch(const ManualMotionDetector* detector, AutoDetectionProvider* detection_provider);

    // Назначает детектор движения для поиска кандидатов.
    void configure(const ManualMotionDetector* detector, AutoDetectionProvider* detection_provider);
    // Обновляет параметры фильтрации автодетектора движения.
    void configure_motion_filter(int iterations,
                                 float diffusion_pixels,
                                 float cluster_ratio_threshold,
                                 int history_size,
                                 int diff_threshold,
                                 double min_area);
    // Сбрасывает активный поиск и очищает буферы кадров.
    void reset();

    // Инициализирует поиск вокруг последней позиции цели.
    void start(const cv::Point2f& last_pos, long long now_ms, const cv::Mat& frame);
    // Продвигает поиск и возвращает найденный bbox при успехе.
    bool update(const cv::Mat& frame, cv::Rect2f& out_bbox);

    // Обновляет список текущих отслеживаемых bbox.
    void set_tracked_boxes(const std::vector<cv::Rect2f>& tracked_boxes);
    void set_reserved_detection_points(const std::vector<cv::Point2f>& reserved_points);

    // Возвращает флаг активного сбора кадров.
    bool active() const { return active_; }

private:
    const ManualMotionDetector* detector_ = nullptr; // - detector_: указатель на детектор движения для поиска кандидатов.
    bool started_ = false; // - started_: был ли поиск инициализирован (зафиксирован last_pos_).
    bool active_ = false; // - active_: выполняется ли активный сбор кадров для ROI.
    bool best_candidate_selected_ = false; // - best_candidate_selected_: выбрана ли ближайшая точка для ROI.
    long long start_ms_ = 0; // - start_ms_: время начала поиска (используется для тайминга снаружи).
    cv::Rect roi_; // - roi_: текущая область, в которой ищется движение.
    std::vector<cv::Mat> gray_frames_; // - gray_frames_: накопленные кадры в ттенках серого для анализа движения.
    cv::Point2f last_pos_{0.0f, 0.0f}; // - last_pos_: последняя известная позиция цели, вокруг которой строится ROI.
    std::vector<cv::Rect2f> tracked_boxes_; // - tracked_boxes_: bbox уже отслеживаемых целей для фильтрации.
    AutomaticMotionDetector automatic_detector_; // - automatic_detector_: детектор движения для поиска кандидатов.
    AutoDetectionProvider* detection_provider_ = nullptr; // - detection_provider_: провайдер фоновых детекций.
};
