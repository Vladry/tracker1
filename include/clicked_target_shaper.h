#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

struct ClickedTargetShaperConfig {
    int CLICK_CAPTURE_SIZE = 80; // - сторона ROI вокруг клика для поиска движения.
    int MOTION_FRAMES = 3; // - количество кадров для анализа движения.
    int MOTION_DIFF_THRESHOLD = 25; // - порог бинаризации diff-кадра.
    int CLICK_PADDING = 6; // - дополнительный паддинг вокруг найденной области движения.
    int TRACKER_INIT_PADDING = 10; // - расширение bbox при старте OpenCV-трекера.
    int TRACKER_MIN_SIZE = 24; // - минимальная сторона bbox для инициализации трекера.
    float MOTION_MIN_MAGNITUDE = 0.4f; // - минимальная средняя длина вектора движения.
    float MOTION_ANGLE_TOLERANCE_DEG = 20.0f; // - допуск по углу движения (в градусах).
    float MOTION_MAG_TOLERANCE_PX = 3.0f; // - допуск по длине шага (в пикселях).
    int MAX_FEATURES = 200; // - максимальное число ключевых точек для goodFeaturesToTrack.
    float QUALITY_LEVEL = 0.01f; // - порог качества для goodFeaturesToTrack.
    float MIN_DISTANCE = 3.0f; // - минимальная дистанция между ключевыми точками.
    float ANGLE_BIN_DEG = 10.0f; // - размер бина направлений (градусы).
    float MAG_BIN_PX = 2.0f; // - размер бина длины шага (пиксели).
    float GRID_STEP_RATIO = 0.1f; // - шаг сетки для проверки стабильности (доля ROI).
    float MIN_STABLE_RATIO = 0.1f; // - доля совпавших пикселей для подтверждения цели.
    int MIN_AREA = 200; // - минимальная площадь ROI для создания кандидата.
    int MIN_WIDTH = 10; // - минимальная ширина ROI.
    int MIN_HEIGHT = 10; // - минимальная высота ROI.
};

// Формирует целевой кластер и bbox после ручного клика.
class ClickedTargetShaper {
public:
    ClickedTargetShaper() = default;
    // Создаёт формирователь цели с заданной конфигурацией.
    explicit ClickedTargetShaper(const ClickedTargetShaperConfig& cfg) : cfg_(cfg) {}

    // Обновляет текущую конфигурацию.
    void update_config(const ClickedTargetShaperConfig& cfg) { cfg_ = cfg; }
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
    ClickedTargetShaperConfig cfg_{}; // - активная конфигурация поиска движения и пороги фильтрации.

    // Строит ROI движения по оптическому потоку между кадрами.
    cv::Rect2f build_motion_roi_from_sequence(const std::vector<cv::Mat>& frames,
                                              const cv::Rect& roi,
                                              std::vector<cv::Point2f>& motion_points) const;
    // Строит ROI движения по разнице первого и последнего кадра.
    cv::Rect2f build_motion_roi_from_diff(const std::vector<cv::Mat>& frames,
                                          const cv::Rect& roi) const;
};
