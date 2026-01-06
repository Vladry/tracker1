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
    void reset();

    void update(const cv::Mat& frame, long long now_ms);
    const std::vector<cv::Point2f>& detections() const { return filtered_points_; }

private:
    static cv::Mat to_gray(const cv::Mat& frame);
    static cv::Point2f rect_center(const cv::Rect& rect);
    std::vector<cv::Point2f> detect_by_motion(const cv::Mat& frame);
    void rebuild_filtered_points();

    std::vector<cv::Mat> gray_history_;
    std::deque<std::vector<cv::Point2f>> detection_history_;
    std::vector<cv::Point2f> filtered_points_;
    int detection_iterations_ = 10;
    float diffusion_pixels_ = 100.0f;
    float cluster_ratio_threshold_ = 0.9f;
    int history_size_ = 5;
    int diff_threshold_ = 25;
    double min_area_ = 60.0;
    int update_period_ms_ = 100;
    long long last_update_ms_ = 0;
};
