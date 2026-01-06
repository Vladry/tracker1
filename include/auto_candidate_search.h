#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "detection_matcher.h"
#include "clicked_target_shaper.h"

class AutoCandidateSearch {
public:
    AutoCandidateSearch() = default;
    // Создаёт поисковый модуль с заданным формирователем цели и фоновым детектором.
    AutoCandidateSearch(const ClickedTargetShaper* detector, MotionDetector* motion_detector);

    // Назначает формирователь цели и фоновый детектор для поиска кандидатов.
    void configure(const ClickedTargetShaper* detector, MotionDetector* motion_detector);
    // Обновляет параметры фильтрации фонового детектора движения.
    void configure_motion_filter(int iterations,
                                 float diffusion_pixels,
                                 float cluster_ratio_threshold,
                                 int history_size,
                                 int diff_threshold,
                                 double min_area);
    // Устанавливает радиус, в котором зарезервированные кандидаты исключаются из выбора.
    void set_reserved_detection_radius(float radius_px);
    // Сбрасывает активный поиск и очищает буферы кадров.
    void reset();

    // Инициализирует поиск вокруг последней позиции цели.
    void start(const cv::Point2f& last_pos, long long now_ms, const cv::Mat& frame);
    // Продвигает поиск и возвращает найденнй bbox при успехе.
    bool update(const cv::Mat& frame, cv::Rect2f& out_bbox);

    // Обновляет список текущих отслеживаемых bbox.
    void set_tracked_boxes(const std::vector<cv::Rect2f>& tracked_boxes);
    void set_reserved_detection_points(const std::vector<cv::Point2f>& reserved_points);

    // Возвращает флаг активного сбора кадров.
    bool active() const { return active_; }

private:
    const ClickedTargetShaper* detector_ = nullptr; // - указатель на формирователь цели по клику.
    bool started_ = false; // - был ли поиск инициализирован (зафиксирован last_pos_).
    bool active_ = false; // - выполняется ли активный сбор кадров для ROI.
    bool best_candidate_selected_ = false; // - выбрана ли ближайшая точка для ROI.
    long long start_ms_ = 0; // - время начала поиска (используется для тайминга снаружи).
    cv::Rect roi_; // - текущая область, в которой ищется движение.
    std::vector<cv::Mat> gray_frames_; // - накопленные кадры в оттенках серого для анализа движения.
    cv::Point2f last_pos_{0.0f, 0.0f}; // - последняя известная позиция цели, вокруг которой строится ROI.
    std::vector<cv::Rect2f> tracked_boxes_; // - bbox уже отслеживаемых целей для фильтрации.
    DetectionMatcher detection_matcher_; // - выбор ближайшей детекции из пула.
    MotionDetector* motion_detector_ = nullptr; // - фоновый детектор движения (пул кандидатов).
};
