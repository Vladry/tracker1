#include "detection_matcher.h"

#include <algorithm>
#include <cmath>
#include <limits>

/*
    Документация по detection_matcher.cpp

    Порядок описания переменных и полей классов (формат "<variable> = value; // описание"):
    detector_ = nullptr;               // Указатель на формирователь цели по клику.
    motion_detector_ = nullptr;        // Фоновый детектор движения с пулом кандидатов.
    tracked_boxes_ = {};               // Прямоугольники уже отслеживаемых целей; исключаются из результатов.
    detection_iterations_ = 10;        // Число итераций, используемых провайдером детекций.
    diffusion_pixels_ = 100.0f;        // Радиус клстеризации точек (в пикселях) при усреднении.
    cluster_ratio_threshold_ = 0.9f;   // Минимальная доля точек кластера от общего числа.

    Порядок (последовательность) вызовов функций при поиске кандидата:
    1) find_best_candidate(cx, cy, out_point)
       - Главная точка входа. Создаёт reference из (cx, cy).
       - Получает набор детекций у фонового детектора.
       - Передаёт точки в find_nearest(...), который учитывает резервирование.

    2) find_nearest(reference, points, out_point)
       - Пропускает точки, попадающие в tracked_boxes_.
       - Выбирает ближайшую к reference точку по евклидову расстоянию.
*/

DetectionMatcher::DetectionMatcher(const ClickedTargetShaper* detector)
        : detector_(detector) {}

void DetectionMatcher::set_detector(const ClickedTargetShaper* detector) {
    detector_ = detector;
}

void DetectionMatcher::set_motion_detector(MotionDetector* motion_detector) {
    motion_detector_ = motion_detector;
}

void DetectionMatcher::set_tracked_boxes(const std::vector<cv::Rect2f>& tracked_boxes) {
    tracked_boxes_ = tracked_boxes;
}

void DetectionMatcher::set_reserved_detection_points(const std::vector<cv::Point2f>& reserved_points) {
    reserved_detection_points_ = reserved_points;
}

void DetectionMatcher::set_reserved_detection_radius(float radius_px) {
    reserved_detection_radius_ = std::max(0.0f, radius_px);
}

void DetectionMatcher::set_detection_params(int iterations,
                                            float diffusion_pixels,
                                            float cluster_ratio_threshold) {
    detection_iterations_ = std::max(1, iterations);
    diffusion_pixels_ = std::max(1.0f, diffusion_pixels);
    cluster_ratio_threshold_ = std::max(0.0f, std::min(cluster_ratio_threshold, 1.0f));
    if (motion_detector_) {
        motion_detector_->set_detection_params(detection_iterations_,
                                               diffusion_pixels_,
                                               cluster_ratio_threshold_);
    }
}

void DetectionMatcher::set_motion_params(int history_size, int diff_threshold, double min_area) {
    history_size_ = std::max(2, history_size);
    diff_threshold_ = std::max(0, diff_threshold);
    min_area_ = std::max(0.0, min_area);
    if (motion_detector_) {
        motion_detector_->set_motion_params(history_size_, diff_threshold_, min_area_);
    }
}

void DetectionMatcher::reset_state() {
    tracked_boxes_.clear();
    reserved_detection_points_.clear();
}

// Выбирает ближайшую детекцию, игнорируя зарезервированные точки в радиусе reserved_detection_radius_.
bool DetectionMatcher::find_nearest(const cv::Point2f& reference, // точка потерянного трека,возле которого ищем новую детекцию
                                    const std::vector<cv::Point2f>& points, // filtered_points_  возвращённые из detections()
                                    cv::Point2f& out_point) const {
    float best_dist = std::numeric_limits<float>::max();
    bool found = false;
    const float reserved_radius_sq = reserved_detection_radius_ * reserved_detection_radius_;

    // пробегаем по точкам авто-детекции и, для каждой смотрим лежит ли она в области bbox какого-то трека:
    int counter = 0;
    for (const auto& point : points) {
        std::cout << std::endl<< " for point: " << counter;
        bool tracked = false;


        for (const auto& rect : tracked_boxes_) {
            if (rect.contains(point)) {
                tracked = true;
                break;
            }
        }
        if (tracked) {
            continue;
        }

        // выкусывание областей точек рядом с недавно "отданной" детекцией предыдущему потерянному треку:
        bool reserved = false;
        for (const auto& reserved_point : reserved_detection_points_) {
            const float dx = point.x - reserved_point.x;
            const float dy = point.y - reserved_point.y;
            if (dx * dx + dy * dy <= reserved_radius_sq) {
                reserved = true;
                break;
            }
        }

        if (reserved) {
            continue;
        }

        const float dx = point.x - reference.x;
        const float dy = point.y - reference.y;
        const float dist = std::sqrt(dx * dx + dy * dy);
        if (dist < best_dist) {
            std::cout << "    best_dist = " << dist;
            best_dist = dist;
            out_point = point;
            found = true;
        }
        counter++;
    }

    return found;
}

bool DetectionMatcher::find_best_candidate(int cx, int cy, cv::Point2f& out_point) const {
    if (!motion_detector_) {
        return false;
    }

    // Берём актуальный пул детекций из MotionDetector и ищем ближайшую к reference.
    const cv::Point2f reference(static_cast<float>(cx), static_cast<float>(cy));
    // просто записывает filtered_points_ в detections
    const std::vector<cv::Point2f>& detections = motion_detector_->detections();
    if (detections.empty()) {
        return false;
    }
    return find_nearest(reference, detections, out_point);
}
