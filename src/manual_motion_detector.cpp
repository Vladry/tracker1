#include "manual_motion_detector.h"
#include <algorithm>
#include <cmath>
#include <map>

namespace {
    // Возвращает минимальный угол между направлениями a и b в радианах.
    // Нужен для проверки стабильности направления движения между кадрами.
    static inline float angle_between(float a, float b) {
        constexpr float kPi = 3.14159265f;
        float diff = std::fabs(a - b);
        if (diff > kPi) {
            diff = 2.0f * kPi - diff;
        }
        return diff;
    }

    // Обрезает прямоугольник до границ кадра.
    // Используется, чтобы не выходить за пределы изображения после расчётов ROI.
    static inline cv::Rect2f clip_rect(const cv::Rect2f& rect, const cv::Size& size) {
        float x1 = std::max(0.0f, rect.x);
        float y1 = std::max(0.0f, rect.y);
        float x2 = std::min(rect.x + rect.width, static_cast<float>(size.width));
        float y2 = std::min(rect.y + rect.height, static_cast<float>(size.height));
        if (x2 <= x1 || y2 <= y1) {
            return {};
        }
        return {x1, y1, x2 - x1, y2 - y1};
    }

    // Расширяет прямоугольник на pad пикселей по всем сторонам.
    // Применяется при создании bbox для инициализации трекера.
    static inline cv::Rect2f expand_rect(const cv::Rect2f& rect, float pad) {
        return cv::Rect2f(rect.x - pad, rect.y - pad, rect.width + pad * 2.0f, rect.height + pad * 2.0f);
    }

    // Гарантирует минимальный размер прямоугольника.
    // Центр сохраняется, чтобы не смещать цель при увеличении.
    static inline cv::Rect2f ensure_min_size(const cv::Rect2f& rect, float min_size) {
        if (rect.width >= min_size && rect.height >= min_size) {
            return rect;
        }
        const cv::Point2f center(rect.x + rect.width * 0.5f, rect.y + rect.height * 0.5f);
        const float size = std::max(min_size, std::max(rect.width, rect.height));
        return cv::Rect2f(center.x - size * 0.5f, center.y - size * 0.5f, size, size);
    }
}

// Возвращает число кадров, необходимых для анализа движения (motion_frames + базовый кадр).
int ManualMotionDetector::required_frames() const {
    return std::max(1, cfg_.motion_frames) + 1;
}

// Формирует ROI вокруг клика, ограничивая его границами кадра.
// Это исходная область, в которой проверяется наличие движения.
cv::Rect ManualMotionDetector::make_click_roi(const cv::Mat& frame, int x, int y) const {
    if (frame.empty()) {
        return {};
    }
    const int size = std::max(2, cfg_.click_capture_size);
    const int half = size / 2;
    const int x1 = std::max(0, x - half);
    const int y1 = std::max(0, y - half);
    const int x2 = std::min(frame.cols, x + half);
    const int y2 = std::min(frame.rows, y + half);
    if (x2 <= x1 || y2 <= y1) {
        return {};
    }
    return {x1, y1, x2 - x1, y2 - y1};
}

// Строит ROI движения на основе оптического потока между кадрами.
// Возвращает bbox движения и заполняет motion_points финальными точками движения.
cv::Rect2f ManualMotionDetector::build_motion_roi_from_sequence(
        const std::vector<cv::Mat>& frames,
        const cv::Rect& roi,
        std::vector<cv::Point2f>& motion_points) const {
    motion_points.clear();
    if (frames.size() < 2 || roi.width <= 0 || roi.height <= 0) {
        return {};
    }

    constexpr int kMaxFeatures = 400;
    constexpr float kQuality = 0.01f;
    constexpr float kMinDistance = 3.0f;
    const float kAngleTolDeg = std::max(1.0f, cfg_.motion_angle_tolerance_deg);
    const float kMagTolPx = std::max(0.1f, cfg_.motion_mag_tolerance_px);
    const float kMinMotionPx = std::max(0.05f, cfg_.motion_min_magnitude);
    constexpr float kAngleBinDeg = 10.0f;
    constexpr float kMagBinPx = 2.0f;

    constexpr float kPi = 3.14159265f;
    const float angle_tol = kAngleTolDeg * kPi / 180.0f;
    const float angle_bin = kAngleBinDeg * kPi / 180.0f;

    const cv::Mat& base = frames.front();
    cv::Mat roi_gray = base(roi);

    std::vector<cv::Point2f> points;
    cv::goodFeaturesToTrack(roi_gray, points, kMaxFeatures, kQuality, kMinDistance);
    if (points.empty()) {
        return {};
    }
    for (auto& p : points) {
        p.x += static_cast<float>(roi.x);
        p.y += static_cast<float>(roi.y);
    }

    const int steps = static_cast<int>(frames.size()) - 1;
    std::vector<cv::Point2f> prev_points = points;
    std::vector<cv::Point2f> curr_points;
    std::vector<std::vector<cv::Point2f>> step_vectors(points.size());
    std::vector<bool> valid(points.size(), true);

    for (size_t i = 1; i < frames.size(); ++i) {
        std::vector<unsigned char> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(frames[i - 1], frames[i], prev_points, curr_points, status, err);
        for (size_t idx = 0; idx < points.size(); ++idx) {
            if (!valid[idx]) {
                continue;
            }
            if (idx >= status.size() || status[idx] == 0) {
                valid[idx] = false;
                continue;
            }
            cv::Point2f step = curr_points[idx] - prev_points[idx];
            step_vectors[idx].push_back(step);
        }
        prev_points = curr_points;
    }

    // Кандидат движения:
    // - last_pos: последняя позиция точки после трекинга.
    // - mean_step: усреднённый вектор движения по всем кадрам.
    struct MotionCandidate {
        cv::Point2f last_pos;
        cv::Point2f mean_step;
    };

    std::vector<MotionCandidate> candidates;
    candidates.reserve(points.size());
    for (size_t idx = 0; idx < points.size(); ++idx) {
        if (!valid[idx] || static_cast<int>(step_vectors[idx].size()) != steps) {
            continue;
        }
        cv::Point2f sum(0.0f, 0.0f);
        for (const auto& step : step_vectors[idx]) {
            sum += step;
        }
        cv::Point2f mean(sum.x / static_cast<float>(steps),
                         sum.y / static_cast<float>(steps));
        const float mean_mag = std::sqrt(mean.x * mean.x + mean.y * mean.y);
        if (mean_mag < kMinMotionPx) {
            continue;
        }
        const float mean_angle = std::atan2(mean.y, mean.x);
        bool ok = true;
        for (const auto& step : step_vectors[idx]) {
            const float mag = std::sqrt(step.x * step.x + step.y * step.y);
            if (mag < kMinMotionPx * 0.25f) {
                ok = false;
                break;
            }
            const float angle = std::atan2(step.y, step.x);
            if (angle_between(angle, mean_angle) > angle_tol) {
                ok = false;
                break;
            }
            if (std::fabs(mag - mean_mag) > kMagTolPx) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            continue;
        }
        candidates.push_back({prev_points[idx], mean});
    }

    if (candidates.empty()) {
        return {};
    }

    std::map<std::pair<int, int>, int> bins;
    int best_count = 0;
    std::pair<int, int> best_bin{0, 0};
    for (const auto& cand : candidates) {
        const float angle = std::atan2(cand.mean_step.y, cand.mean_step.x);
        const float mag = std::sqrt(cand.mean_step.x * cand.mean_step.x +
                                    cand.mean_step.y * cand.mean_step.y);
        const int angle_key = static_cast<int>(std::round(angle / angle_bin));
        const int mag_key = static_cast<int>(std::round(mag / kMagBinPx));
        const std::pair<int, int> key{angle_key, mag_key};
        const int count = ++bins[key];
        if (count > best_count) {
            best_count = count;
            best_bin = key;
        }
    }

    const float best_angle = static_cast<float>(best_bin.first) * angle_bin;
    const float best_mag = static_cast<float>(best_bin.second) * kMagBinPx;
    std::vector<cv::Point2f> selected;
    selected.reserve(candidates.size());
    for (const auto& cand : candidates) {
        const float angle = std::atan2(cand.mean_step.y, cand.mean_step.x);
        const float mag = std::sqrt(cand.mean_step.x * cand.mean_step.x +
                                    cand.mean_step.y * cand.mean_step.y);
        if (angle_between(angle, best_angle) <= angle_tol &&
            std::fabs(mag - best_mag) <= kMagTolPx) {
            selected.push_back(cand.last_pos);
        }
    }

    if (selected.empty()) {
        return {};
    }

    motion_points = selected;
    cv::Rect rect = cv::boundingRect(selected);
    return clip_rect(cv::Rect2f(rect), base.size());
}

// Строит ROI движения по разнице первого и последнего кадров.
// Используется как запасной путь, если оптический поток не дал результата.
cv::Rect2f ManualMotionDetector::build_motion_roi_from_diff(
        const std::vector<cv::Mat>& frames,
        const cv::Rect& roi) const {
    if (frames.size() < 2 || roi.width <= 0 || roi.height <= 0) {
        return {};
    }
    const cv::Mat& first = frames.front();
    const cv::Mat& last = frames.back();
    cv::Mat diff;
    cv::absdiff(first, last, diff);
    cv::Mat roi_diff = diff(roi);
    cv::threshold(roi_diff, roi_diff, cfg_.motion_diff_threshold, 255, cv::THRESH_BINARY);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(roi_diff, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return {};
    }
    cv::Rect best_rect;
    double best_area = 0.0;
    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < cfg_.min_area * 0.5) {
            continue;
        }
        if (area > best_area) {
            best_area = area;
            best_rect = cv::boundingRect(contour);
        }
    }
    if (best_rect.area() <= 0) {
        return {};
    }
    best_rect.x += roi.x;
    best_rect.y += roi.y;
    return clip_rect(cv::Rect2f(best_rect), first.size());
}

// Пытается построить кандидата трека по серии кадров.
// Возвращает true, если найден пригодный bbox для трекера.
// out_tracker_roi — bbox для инициализации трекера,
// motion_points — опциональные точки движения,
// motion_roi_out — опциональный bbox движения до паддинга.
bool ManualMotionDetector::build_candidate(
        const std::vector<cv::Mat>& gray_frames,
        const cv::Rect& roi,
        const cv::Size& frame_size,
        cv::Rect2f& out_tracker_roi,
        std::vector<cv::Point2f>* motion_points,
        cv::Rect2f* motion_roi_out) const {
    if (gray_frames.size() < 2 || roi.area() <= 0) {
        return false;
    }

    std::vector<cv::Point2f> local_points;
    cv::Rect2f motion_roi = build_motion_roi_from_sequence(gray_frames, roi, local_points);
    if (motion_roi.area() <= 1.0f) {
        local_points.clear();
        motion_roi = build_motion_roi_from_diff(gray_frames, roi);
    }

    if (motion_roi.area() > 1.0f) {
        motion_roi.x -= cfg_.click_padding;
        motion_roi.y -= cfg_.click_padding;
        motion_roi.width += cfg_.click_padding * 2.0f;
        motion_roi.height += cfg_.click_padding * 2.0f;
        motion_roi = clip_rect(motion_roi, frame_size);
    }

    if (motion_roi_out) {
        *motion_roi_out = motion_roi;
    }

    if (motion_roi.area() > 1.0f &&
        motion_roi.area() >= static_cast<float>(cfg_.min_area) &&
        motion_roi.width >= cfg_.min_width &&
        motion_roi.height >= cfg_.min_height) {
        cv::Rect2f tracker_roi = motion_roi;
        tracker_roi = expand_rect(tracker_roi, static_cast<float>(cfg_.tracker_init_padding));
        tracker_roi = ensure_min_size(tracker_roi, static_cast<float>(cfg_.tracker_min_size));
        tracker_roi = clip_rect(tracker_roi, frame_size);
        if (tracker_roi.area() <= 1.0f) {
            return false;
        }
        out_tracker_roi = tracker_roi;
        if (motion_points) {
            *motion_points = std::move(local_points);
        }
        return true;
    }

    return false;
}
