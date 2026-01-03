#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

struct ManualMotionDetectorConfig {
    int click_capture_size = 80; // - click_capture_size: сторона ROI вокруг клика для поиска движения.
    int motion_frames = 3; // - motion_frames: количество кадров для анализа движения.
    int motion_diff_threshold = 25; // - motion_diff_threshold: порог бинаризации diff-кадра.
    int click_padding = 6; // - click_padding: дополнительный паддинг вокруг найденной области движения.
    int tracker_init_padding = 10; // - tracker_init_padding: расширение bbox при старте OpenCV-трекера.
    int tracker_min_size = 24; // - tracker_min_size: минимальная сторона bbox для инициализации трекера.
    float motion_min_magnitude = 0.4f; // - motion_min_magnitude: минимальная средняя длина вектора движения.
    float motion_angle_tolerance_deg = 20.0f; // - motion_angle_tolerance_deg: допуск по углу движения (в градусах).
    float motion_mag_tolerance_px = 3.0f; // - motion_mag_tolerance_px: допуск по длине шага (в пикселях).
    int min_area = 200; // - min_area: минимальная площадь ROI для создания кандидата.
    int min_width = 10; // - min_width: минимальная ширина ROI.
    int min_height = 10; // - min_height: минимальная высота ROI.
};

class ManualMotionDetector {
public:
    ManualMotionDetector() = default;
    // Создаёт детектор движения с заданной конфигурацией.
    explicit ManualMotionDetector(const ManualMotionDetectorConfig& cfg) : cfg_(cfg) {}

    // Обновляет текущую конфигурацию детектора.
    void update_config(const ManualMotionDetectorConfig& cfg) { cfg_ = cfg; }
    // Возвращает число кадров, требуемых для анализа движения.
    int required_frames() const;

    // Формирует ROI вокруг клика внутри границ кадра.
    cv::Rect make_click_roi(const cv::Mat& frame, int x, int y) const;

    // Строит bbox кандидата трека на основе последовательности серых кадров.
    bool build_candidate(const std::vector<cv::Mat>& gray_frames,
                         const cv::Rect& roi,
                         const cv::Size& frame_size,
                         cv::Rect2f& out_tracker_roi,
                         std::vector<cv::Point2f>* motion_points,
                         cv::Rect2f* motion_roi_out) const;

private:
    ManualMotionDetectorConfig cfg_{}; // - cfg_: активная конфигурация поиска движения и пороги фильтрации.

    // Строит ROI движения по оптическому потоку между кадрами.
    cv::Rect2f build_motion_roi_from_sequence(const std::vector<cv::Mat>& frames,
                                              const cv::Rect& roi,
                                              std::vector<cv::Point2f>& motion_points) const;
    // Строит ROI движения по разнице первого и последнего кадра.
    cv::Rect2f build_motion_roi_from_diff(const std::vector<cv::Mat>& frames,
                                          const cv::Rect& roi) const;
};
