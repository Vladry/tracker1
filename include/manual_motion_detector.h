#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

struct ManualMotionDetectorConfig {
    int CLICK_CAPTURE_SIZE = 80; // - CLICK_CAPTURE_SIZE: сторона ROI вокруг клика для поиска движения.
    int MOTION_FRAMES = 3; // - MOTION_FRAMES: количество кадров для анализа движения.
    int MOTION_DIFF_THRESHOLD = 25; // - MOTION_DIFF_THRESHOLD: порог бинаризации diff-кадра.
    int CLICK_PADDING = 6; // - CLICK_PADDING: дополнительный паддинг вокруг найденной области движения.
    int TRACKER_INIT_PADDING = 10; // - TRACKER_INIT_PADDING: расширение bbox при старте OpenCV-трекера.
    int TRACKER_MIN_SIZE = 24; // - TRACKER_MIN_SIZE: минимальная сторона bbox для инициализации трекера.
    float MOTION_MIN_MAGNITUDE = 0.4f; // - MOTION_MIN_MAGNITUDE: минимальная средняя длина вектора движения.
    float MOTION_ANGLE_TOLERANCE_DEG = 20.0f; // - MOTION_ANGLE_TOLERANCE_DEG: допуск по углу движения (в градусах).
    float MOTION_MAG_TOLERANCE_PX = 3.0f; // - MOTION_MAG_TOLERANCE_PX: допуск по длине шага (в пикселях).
    int MAX_FEATURES = 200; // - MAX_FEATURES: максимальное число ключевых точек для goodFeaturesToTrack.
    float QUALITY_LEVEL = 0.01f; // - QUALITY_LEVEL: порог качества для goodFeaturesToTrack.
    float MIN_DISTANCE = 3.0f; // - MIN_DISTANCE: минимальная дистанция между ключевыми точками.
    float ANGLE_BIN_DEG = 10.0f; // - ANGLE_BIN_DEG: размер бина направлений (градусы).
    float MAG_BIN_PX = 2.0f; // - MAG_BIN_PX: размер бина длины шага (пиксели).
    float GRID_STEP_RATIO = 0.1f; // - GRID_STEP_RATIO: шаг сетки для проверки стабильности (доля ROI).
    float MIN_STABLE_RATIO = 0.1f; // - MIN_STABLE_RATIO: доля совпавших пикселей для подтверждения цели.
    int MIN_AREA = 200; // - MIN_AREA: минимальная площадь ROI для создания кандидата.
    int MIN_WIDTH = 10; // - MIN_WIDTH: минимальная ширина ROI.
    int MIN_HEIGHT = 10; // - MIN_HEIGHT: минимальная высота ROI.
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
