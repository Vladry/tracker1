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

    constexpr int kMaxFeatures = 200;             // kMaxFeatures: максимальное число ключевых точек (features),
    //              которые будем трекать. Больше — лучше покрытие,
    //              но дороже по вычислениям.
    constexpr float kQuality = 0.01f;             // kQuality: порог качества для goodFeaturesToTrack.
    //            Меньше — больше точек, но потенциально шумнее.
    constexpr float kMinDistance = 3.0f;          // kMinDistance: минимальная дистанция между найденными точками.
    //              Предотвращает "слипание" features.
    const float kAngleTolDeg = std::max(1.0f, cfg_.motion_angle_tolerance_deg);
    // kAngleTolDeg: допуск угла движения (в градусах),
    //               взят из конфигурации, но не меньше 1.0.
    //               Определяет, насколько строго считаем
    //               наравление движения согласованным.
    const float kMagTolPx = std::max(0.1f, cfg_.motion_mag_tolerance_px);
    // kMagTolPx: допуск по длине шага (в пикселях),
    //            из конфигурации. Меньше 0.1 не даём,
    //            чтобы не было слишком жёсткой фильтрации.
    const float kMinMotionPx = std::max(0.05f, cfg_.motion_min_magnitude);
    // kMinMotionPx: минимальная средняя длина вектора движения.
    //              Фильтрует почти неподвижные точки (шум).
    constexpr float kAngleBinDeg = 10.0f;         // kAngleBinDeg: размер бина для кластеризации направлений (в градусах).
    //                Чем меньше — тем тоньше кластеризация.
    constexpr float kMagBinPx = 2.0f;             // kMagBinPx: размер бина для кластеризации длины шага (в пикселях).
    //            Чем меньше — тем точнее, но больше кластеров.

    constexpr float kPi = 3.14159265f;            // kPi: π, используется для перевода градусов в радианы.
    const float angle_tol = kAngleTolDeg * kPi / 180.0f;
    // angle_tol: допуск угла в радианах,
    //            используется в angle_between().
    const float angle_bin = kAngleBinDeg * kPi / 180.0f;
    // angle_bin: ширина бина угла в радианах для кластеризации.

    const cv::Mat& base = frames.front();         // base: первый кадр — база для поиска начальных точек.
    cv::Mat roi_gray = base(roi);                 // roi_gray: часть кадра в пределах ROI,
    //            только её анализируем.

    std::vector<cv::Point2f> points;              // points: исходные точки (features) в ROI.
    cv::goodFeaturesToTrack(roi_gray, points, kMaxFeatures, kQuality, kMinDistance);
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
        const int mag_key = static_cast<int>(std::round(mag / kMagBinPx));
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
    const float best_mag = static_cast<float>(best_bin.second) * kMagBinPx;
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
        return {};                                // Нет выбранных точек — движения нет.
    }

    motion_points = selected;                     // Возвращаем точки движения наружу.
    cv::Rect rect = cv::boundingRect(selected);   // rect: ограничивающий bbox по выбранным точкам.
    return clip_rect(cv::Rect2f(rect), base.size());
    // Возвращаем bbox движения, обрезанный по кадру.
}




// Строит ROI движения по разнице первого и последнего кадров.
// Используется как запасной путь, если оптический поток не дал результата.
/*cv::Rect2f ManualMotionDetector::build_motion_roi_from_diff(
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
}*/

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
/*    if (motion_roi.area() <= 1.0f) {
        local_points.clear();
        motion_roi = build_motion_roi_from_diff(gray_frames, roi);
    }*/

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
