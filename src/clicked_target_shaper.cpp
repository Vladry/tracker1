#include "clicked_target_shaper.h"
#include <algorithm>
#include <cmath>
#include <map>

namespace {
    // Возвращает минимальный угол между направлениями a и b в радианах.
    // Нужен для проверки стабильности направления движения между кадрми.
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
    // Центр сохранятся, чтобы не смещать цель при увеличении.
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
int ClickedTargetShaper::required_frames() const {
    return std::max(1, cfg_.MOTION_FRAMES) + 1;
}

// Формирует ROI вокруг клика, ограничивая его границами кадра.
// Это исходная область, в которой проверяется наличие движения.
cv::Rect ClickedTargetShaper::make_click_roi(const cv::Mat& frame, int x, int y) const {
    if (frame.empty()) {
        return {};
    }
    const int size = std::max(2, cfg_.CLICK_CAPTURE_SIZE);
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
cv::Rect2f ClickedTargetShaper::build_motion_roi_from_sequence(
        const std::vector<cv::Mat>& frames,      // frames: последовательность серых кадров для анализа движения.
        //          Чем больше кадров, тем стабильнее оценка движения,
        //          но тем выше задержка и стоимость вычислений.
        const cv::Rect& roi,                      // roi: область интереса вокруг клика, где ищем движение.
        //      Ограничивает поиск, чтобы не анализировать весь кадр.
        std::vector<cv::Point2f>& motion_points   // motion_points: выходной массив точек, относящихся к
        //                согласованному движению (последние позиции).
) const {
    motion_points.clear();                        // motion_points.clear(): сбрасываем предыдущий результат.
    if (frames.size() < 2 || roi.width <= 0 || roi.height <= 0) {
        return {};                                // Если мало кадров или ROI некорректен, движения не определить.
    }

    const int max_features = std::max(1, cfg_.MAX_FEATURES);             // max_features: максимум ключевых точек.
    const float quality = std::max(0.0f, cfg_.QUALITY_LEVEL);            // quality: порог качества goodFeaturesToTrack.
    const float min_distance = std::max(0.0f, cfg_.MIN_DISTANCE);         // min_distance: минимум дистанции между точками.
    const float kAngleTolDeg = std::max(1.0f, cfg_.MOTION_ANGLE_TOLERANCE_DEG);
    // kAngleTolDeg: допуск угла движения (в градусах),
    //               взят из конфигурации, но не меньше 1.0.
    //               Определяет, насколько строго считаем
    //               наравление движения согласованным.
    const float kMagTolPx = std::max(0.1f, cfg_.MOTION_MAG_TOLERANCE_PX);
    // kMagTolPx: допуск по длине шага (в пикселях),
    //            из конфигурации. Меньше 0.1 не даём,
    //            чтобы не было слишком жёсткой фильтрации.
    const float kMinMotionPx = std::max(0.05f, cfg_.MOTION_MIN_MAGNITUDE);
    // kMinMotionPx: минимальная средняя длина вектора движения.
    //              Фильтрует почти неподвижные точки (шум).
    const float angle_bin_deg = std::max(0.1f, cfg_.ANGLE_BIN_DEG);       // angle_bin_deg: размер бина направлений.
    const float mag_bin_px = std::max(0.1f, cfg_.MAG_BIN_PX);             // mag_bin_px: размер бина длины шага.

    constexpr float kPi = 3.14159265f;            // kPi: π, используется для перевода градусов в радианы.
    const float angle_tol = kAngleTolDeg * kPi / 180.0f;
    // angle_tol: допуск угла в радианах,
    //            используется в angle_between().
    const float angle_bin = angle_bin_deg * kPi / 180.0f;
    // angle_bin: ширина бина угла в радианах для кластеризации.

    const cv::Mat& base = frames.front();         // base: первый кадр — база для поиска начальных точек.
    cv::Mat roi_gray = base(roi);                 // roi_gray: часть кадра в пределах ROI,
    //            только её анализируем.

    std::vector<cv::Point2f> points;              // points: исходные точки (features) в ROI.
    cv::goodFeaturesToTrack(roi_gray, points, max_features, quality, min_distance);
    // Находит сильные углы/текстуры в ROI.
    if (points.empty()) {
        return {};                                // Нет точек → нет данных для движения.
    }
    for (auto& p : points) {                      // Переводим координаты из ROI в координаты кадра.
        p.x += static_cast<float>(roi.x);
        p.y += static_cast<float>(roi.y);
    }

    const int steps = static_cast<int>(frames.size()) - 1;
    // steps: число шагов трекинга (межкадровых переходов).
    //        Если N кадров → N-1 переход.
    std::vector<cv::Point2f> prev_points = points;// prev_points: точки на предыдущем кадре.
    std::vector<cv::Point2f> curr_points;         // curr_points: точки на текущем кадре (после LK).
    std::vector<std::vector<cv::Point2f>> step_vectors(points.size());
    // step_vectors: для каждой точки сохраняем векторы движения
    //               по шагам (мжкадровые перемещения).
    std::vector<bool> valid(points.size(), true); // valid: флаг, что точка успешно трекалась на всех шагах.

    for (size_t i = 1; i < frames.size(); ++i) {
        std::vector<unsigned char> status;        // status: флаги успешности трекинга каждой точки.
        std::vector<float> err;                   // err: ошибка трекинга (OpenCV LK).
        cv::calcOpticalFlowPyrLK(frames[i - 1], frames[i], prev_points, curr_points, status, err);
        for (size_t idx = 0; idx < points.size(); ++idx) {
            if (!valid[idx]) {
                continue;                         // Уже невалидные точки не обрабатываем.
            }
            if (idx >= status.size() || status[idx] == 0) {
                valid[idx] = false;               // Точка потеряна на этом шаге → исключаем.
                continue;
            }
            cv::Point2f step = curr_points[idx] - prev_points[idx];
            // step: вектор движения точки между кадрами.
            step_vectors[idx].push_back(step);    // Сохраняем шаг для последующей фильтрации.
        }
        prev_points = curr_points;                // Смещаемся по цепочке кадров.
    }

    struct MotionCandidate {
        cv::Point2f last_pos;                     // last_pos: финальная позиция точки после трекинга.
        cv::Point2f mean_step;                    // mean_step: средний вектор движения по всем кадрам.
    };

    std::vector<MotionCandidate> candidates;      // candidates: точки, прошедшие фильтрацию по
    //             направлению и длине шага.
    candidates.reserve(points.size());            // reserve: оптимизация по памяти.
    for (size_t idx = 0; idx < points.size(); ++idx) {
        if (!valid[idx] || static_cast<int>(step_vectors[idx].size()) != steps) {
            continue;                             // Отбрасываем точки, где есть пропуски в траектории.
        }
        cv::Point2f sum(0.0f, 0.0f);              // sum: сумма всех шагов, чтобы получить среднее.
        for (const auto& step : step_vectors[idx]) {
            sum += step;
        }
        cv::Point2f mean(sum.x / static_cast<float>(steps),
                         sum.y / static_cast<float>(steps));
        // mean: средний вектор движения.
        const float mean_mag = std::sqrt(mean.x * mean.x + mean.y * mean.y);
        // mean_mag: длина среднего вектора, критерий "движения".
        if (mean_mag < kMinMotionPx) {
            continue;                             // Слишком слабое движение — исключаем.
        }
        const float mean_angle = std::atan2(mean.y, mean.x);
        // mean_angle: направление среднего движения.
        bool ok = true;                           // ok: флаг прохождения проверки согласованности.
        for (const auto& step : step_vectors[idx]) {
            const float mag = std::sqrt(step.x * step.x + step.y * step.y);
            if (mag < kMinMotionPx * 0.25f) {
                ok = false;                       // Слишком маленькие шаги → шум.
                break;
            }
            const float angle = std::atan2(step.y, step.x);
            if (angle_between(angle, mean_angle) > angle_tol) {
                ok = false;                       // Слишком большой разброс направлений.
                break;
            }
            if (std::fabs(mag - mean_mag) > kMagTolPx) {
                ok = false;                       // Слишком большой разброс по длине шагов.
                break;
            }
        }
        if (!ok) {
            continue;
        }
        candidates.push_back({prev_points[idx], mean});
        // Сохраняем кандидата: финальная позиция и среднее движение.
    }

    if (candidates.empty()) {
        return {};                                // Если нет кандидатов — нет движения.
    }

    std::map<std::pair<int, int>, int> bins;      // bins: счётчики для кластеризации по (угол, длина).
    int best_count = 0;                           // best_count: количество точек в лучшем кластере.
    std::pair<int, int> best_bin{0, 0};           // best_bin: индекс лучшего кластера.
    for (const auto& cand : candidates) {
        const float angle = std::atan2(cand.mean_step.y, cand.mean_step.x);
        const float mag = std::sqrt(cand.mean_step.x * cand.mean_step.x +
                                    cand.mean_step.y * cand.mean_step.y);
        const int angle_key = static_cast<int>(std::round(angle / angle_bin));
        // angle_key: индекс бина направления.
        const int mag_key = static_cast<int>(std::round(mag / mag_bin_px));
        // mag_key: индекс бина длины.
        const std::pair<int, int> key{angle_key, mag_key};
        const int count = ++bins[key];            // count: число кандидатов в этом бине.
        if (count > best_count) {
            best_count = count;
            best_bin = key;                       // Запоминаем самый плотный кластер.
        }
    }

    const float best_angle = static_cast<float>(best_bin.first) * angle_bin;
    // best_angle: центр лучшего углового кластера (радианы).
    const float best_mag = static_cast<float>(best_bin.second) * mag_bin_px;
    // best_mag: центр кластера по длине шага (пиксели).
    std::vector<cv::Point2f> selected;            // selected: итоговые точки, совпадающие с лучшим кластером.
    selected.reserve(candidates.size());
    for (const auto& cand : candidates) {
        const float angle = std::atan2(cand.mean_step.y, cand.mean_step.x);
        const float mag = std::sqrt(cand.mean_step.x * cand.mean_step.x +
                                    cand.mean_step.y * cand.mean_step.y);
        if (angle_between(angle, best_angle) <= angle_tol &&
            std::fabs(mag - best_mag) <= kMagTolPx) {
            selected.push_back(cand.last_pos);    // Отбираем точки согласованного движения.
        }
    }

    if (selected.empty()) {
        // Нет выбранных точек — пытаемся проверить стабильность объекта по разрежённой матрице.
        const float grid_step_ratio = std::max(0.01f, cfg_.GRID_STEP_RATIO);
        const float min_stable_ratio = std::max(0.0f, std::min(cfg_.MIN_STABLE_RATIO, 1.0f));
        const cv::Mat& prev_frame = frames[frames.size() - 2];
        const cv::Mat& curr_frame = frames.back();
        if (prev_frame.size() != curr_frame.size()) {
            return {};                            // Несовпадение размеров кадров.
        }
        const int step_x = std::max(1, static_cast<int>(std::round(roi.width * grid_step_ratio)));
        const int step_y = std::max(1, static_cast<int>(std::round(roi.height * grid_step_ratio)));
        const int start_x = roi.x + step_x / 2;
        const int start_y = roi.y + step_y / 2;
        int total_samples = 0;
        int stable_samples = 0;
        std::vector<cv::Point2f> stable_points;
        for (int y = start_y; y < roi.y + roi.height; y += step_y) {
            for (int x = start_x; x < roi.x + roi.width; x += step_x) {
                if (x < 0 || y < 0 || x >= curr_frame.cols || y >= curr_frame.rows) {
                    continue;
                }
                const int prev_value = prev_frame.at<unsigned char>(y, x);
                const int curr_value = curr_frame.at<unsigned char>(y, x);
                const int diff = std::abs(curr_value - prev_value);
                ++total_samples;
                if (diff <= cfg_.MOTION_DIFF_THRESHOLD) {
                    ++stable_samples;
                    stable_points.emplace_back(static_cast<float>(x),
                                               static_cast<float>(y));
                }
            }
        }
        if (total_samples == 0) {
            return {};
        }
        const float stable_ratio = static_cast<float>(stable_samples) /
                                   static_cast<float>(total_samples);
        if (stable_ratio >= min_stable_ratio) {
            motion_points = std::move(stable_points);
            return clip_rect(cv::Rect2f(roi), base.size());
        }
        return {};                                // Нет подтверждения стабильности.
    }

    motion_points = selected;                     // Возвращаем точки движения наружу.
    cv::Rect rect = cv::boundingRect(selected);   // rect: ограничивающий bbox по выбранным точкам.
    return clip_rect(cv::Rect2f(rect), base.size());
    // Возвращаем bbox движения, обрезанный по кадру.
}

// Цепочка 1 (клик оператора):
// 1) receive gray_frames + ROI от клика,
// 2) build_motion_roi_from_sequence(...) выделяет движущийся кластер пикселей,
// 3) motion_roi расширяется паддингом и проверяется на минимальные размеры,
// 4) итоговый bbox используется для инициализации трекера.
// Возвращает true, если найден пригодный bbox.
bool ClickedTargetShaper::build_candidate(
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
/*    if (motion_roi.area() <= 1.0f) {
        local_points.clear();
        motion_roi = build_motion_roi_from_diff(gray_frames, roi);
    }*/

    if (motion_roi.area() > 1.0f) {
        motion_roi.x -= cfg_.CLICK_PADDING;
        motion_roi.y -= cfg_.CLICK_PADDING;
        motion_roi.width += cfg_.CLICK_PADDING * 2.0f;
        motion_roi.height += cfg_.CLICK_PADDING * 2.0f;
        motion_roi = clip_rect(motion_roi, frame_size);
    }

    if (motion_roi_out) {
        *motion_roi_out = motion_roi;
    }

    if (motion_roi.area() > 1.0f &&
        motion_roi.area() >= static_cast<float>(cfg_.MIN_AREA) &&
        motion_roi.width >= cfg_.MIN_WIDTH &&
        motion_roi.height >= cfg_.MIN_HEIGHT) {
        cv::Rect2f tracker_roi = motion_roi;
        tracker_roi = expand_rect(tracker_roi, static_cast<float>(cfg_.TRACKER_INIT_PADDING));
        tracker_roi = ensure_min_size(tracker_roi, static_cast<float>(cfg_.TRACKER_MIN_SIZE));
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
