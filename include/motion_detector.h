#pragma once

#include <deque>
#include <opencv2/opencv.hpp>
#include <vector>

// Фоновый детектор движения, заполняющий пул кандидатов.
class MotionDetector {
public:
    MotionDetector() = default;

    void set_detection_params(int iterations, float diffusion_pixels, float cluster_ratio_threshold);
    void set_motion_params(int history_size, int diff_threshold, double min_area);
    void set_update_period_ms(int period_ms);
    void set_binarize_max_value(int max_value);
    void reset();

    void update(const cv::Mat& frame, long long now_ms);
    const std::vector<cv::Point2f>& detections() const { return filtered_points_; }

private:
    static cv::Mat to_gray(const cv::Mat& frame);
    static cv::Point2f rect_center(const cv::Rect& rect);
    std::vector<cv::Point2f> detect_by_motion(const cv::Mat& frame);
    void rebuild_filtered_points();

    std::vector<cv::Mat> gray_history_; // - история серых кадров для diff-детекции.
    std::deque<std::vector<cv::Point2f>> detection_history_; // - история наборов кандидатов по кадрам.
    std::vector<cv::Point2f> filtered_points_; // - итоговые отфильтрованные точки-кандидаты.
    int detection_iterations_ = 3; // История кадров для автодетекции движения. История используется при rebuild_filtered_points() — там все точки из истории агрегируются и кластеризуются, чтобы сгладить шум и стабилизировать точки детекций.
    float diffusion_pixels_ = 100.0f; // - радиус кластеризации детекций при агрегации.
    float cluster_ratio_threshold_ = 0.8f; // это порог доли точек, которые должны попасть в один кластер, чтобы этот кластер считался устойчивой детекцией и был добавлен в filtered_points_.
    int history_size_ = 5;  // это размер истории gray_frames кадров. Он используется для формирования diff‑кадра между первым и последним кадром в истории (gray_history_), на основе которого затем вычисляются контуры движения.
    int diff_threshold_ = 15; // - порог бинаризации diff-кадра.
    int binarize_max_value_ = 255; // - максимальное значение пикселя при бинаризации diff-кадра.
    double min_area_ = 60.0; // - минимальная площадь контура, считаемого движением.
    int update_period_ms_ = 100; // - период обновления детекций движения.
    long long last_update_ms_ = 0; // - время последнего обновления пула кандидатов.
};
