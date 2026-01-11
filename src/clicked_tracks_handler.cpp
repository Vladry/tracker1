#include "clicked_tracks_handler.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <string>

namespace {
    // Возвращает центр прямоугольника.
    // Используется для запоминания последней позиции цели.
    static inline cv::Point2f rect_center(const cv::Rect2f &rect) {
        return {rect.x + rect.width * 0.5f, rect.y + rect.height * 0.5f};
    }

    // Переводит прямоугольник с float координатами в int, округляя границы.
    // Нужен для отрисовки и операций с матрицами масок.
    static inline cv::Rect to_int_rect(const cv::Rect2f &rect) {
        return cv::Rect(
                static_cast<int>(std::round(rect.x)),
                static_cast<int>(std::round(rect.y)),
                static_cast<int>(std::round(rect.width)),
                static_cast<int>(std::round(rect.height))
        );
    }

    // Переводит кадр в градации серого, если он в формате BGR.
    // Клонирует кадр, чтобы не модифицировать исходные данные.
    static inline cv::Mat to_gray(const cv::Mat &frame) {
        if (frame.channels() == 3) {
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            return gray;
        }
        return frame.clone();
    }

    // Обрезает прямоугольник по размеру кадра, исключая отрицательные координаты.
    // Гарантирует корректный bbox перед передчей трекеру или рендереру.
    static inline cv::Rect2f clip_rect(const cv::Rect2f &rect, const cv::Size &size) {
        float x1 = std::max(0.0f, rect.x);
        float y1 = std::max(0.0f, rect.y);
        float x2 = std::min(rect.x + rect.width, static_cast<float>(size.width));
        float y2 = std::min(rect.y + rect.height, static_cast<float>(size.height));
        if (x2 <= x1 || y2 <= y1) {
            return {};
        }
        return {x1, y1, x2 - x1, y2 - y1};
    }

}

/*   clicked_tracks_handler :
Принимает клики оператора и создаёт цели из ROI (цепочка ручного клика).
Ведёт треки, обновляет их, переводит в lost, пересоздаёт трекер, ведёт history и т.д.
Запускает авто‑поиск кандидатов (auto‑candidate search) для серых треков.
Использует фоновую автодетекцию для возврата целей (MotionDetector + DetectionMatcher).
То есть это менеджер системы “ручных треков + их авто‑восстановление”.
Он управляет не только “кликнутыми целями”, но и автоматическим восстановлением, резервацией кандидатов и историей видимости.
 */


// Конструктор: поднимает логирование и загружает настройки ручного трекера.
ClickedTracksHandler::ClickedTracksHandler(const toml::table &tbl) {
    load_logging_config(tbl, log_cfg_);
    load_config(tbl);
    ClickedTargetShaperConfig motion_cfg;
    motion_cfg.CLICK_CAPTURE_SIZE = cfg_.CLICK_CAPTURE_SIZE;
    motion_cfg.MOTION_FRAMES = cfg_.MOTION_FRAMES;
    motion_cfg.MOTION_DIFF_THRESHOLD = cfg_.MOTION_DIFF_THRESHOLD;
    motion_cfg.CLICK_PADDING = cfg_.CLICK_PADDING;
    motion_cfg.TRACKER_INIT_PADDING = cfg_.TRACKER_INIT_PADDING;
    motion_cfg.TRACKER_MIN_SIZE = cfg_.TRACKER_MIN_SIZE;
    motion_cfg.MOTION_MIN_MAGNITUDE = cfg_.MOTION_MIN_MAGNITUDE;
    motion_cfg.MOTION_ANGLE_TOLERANCE_DEG = cfg_.MOTION_ANGLE_TOLERANCE_DEG;
    motion_cfg.MOTION_MAG_TOLERANCE_PX = cfg_.MOTION_MAG_TOLERANCE_PX;
    motion_cfg.MAX_FEATURES = cfg_.MOTION_MAX_FEATURES;
    motion_cfg.QUALITY_LEVEL = cfg_.MOTION_QUALITY_LEVEL;
    motion_cfg.MIN_DISTANCE = cfg_.MOTION_MIN_DISTANCE;
    motion_cfg.ANGLE_BIN_DEG = cfg_.MOTION_ANGLE_BIN_DEG;
    motion_cfg.MAG_BIN_PX = cfg_.MOTION_MAG_BIN_PX;
    motion_cfg.GRID_STEP_RATIO = cfg_.MOTION_GRID_STEP_RATIO;
    motion_cfg.MIN_STABLE_RATIO = cfg_.MOTION_MIN_STABLE_RATIO;
    motion_cfg.MIN_AREA = cfg_.MIN_AREA;
    motion_cfg.MIN_WIDTH = cfg_.MIN_WIDTH;
    motion_cfg.MIN_HEIGHT = cfg_.MIN_HEIGHT;
    clicked_target_shaper_.update_config(motion_cfg);
    motion_detector_.set_detection_params(cfg_.MOTION_DETECTION_ITERATIONS,
                                          cfg_.MOTION_DETECTION_DIFFUSION_PX,
                                          cfg_.MOTION_DETECTION_CLUSTER_RATIO);
    motion_detector_.set_motion_params(cfg_.AUTO_HISTORY_SIZE,
                                       cfg_.AUTO_DIFF_THRESHOLD,
                                       cfg_.AUTO_MIN_AREA);
    motion_detector_.set_update_period_ms(cfg_.AUTO_DETECTION_PERIOD_MS);
    motion_detector_.set_binarize_max_value(cfg_.MOTION_BINARY_MAX_VALUE);
}

// Згружает параметры ручного трекера из TOML и печатает краткую сводку.
// Любая ошибка в таблице [manual_tracker] приводит к исключению и сообщению в лог.
bool ClickedTracksHandler::load_config(const toml::table &tbl) {
    try {
        const auto *cfg = tbl["manual_tracker"].as_table();
        if (!cfg) {
            throw std::runtime_error("missing [manual_tracker] table");
        }
        cfg_.MAX_TARGETS = read_required<int>(*cfg, "MAX_TARGETS");
        cfg_.CLICK_PADDING = read_required<int>(*cfg, "CLICK_PADDING");
        cfg_.MOTION_DIFF_THRESHOLD = read_required<int>(*cfg, "MOTION_DIFF_THRESHOLD");
        cfg_.CLICK_CAPTURE_SIZE = read_required<int>(*cfg, "CLICK_CAPTURE_SIZE");
        cfg_.MOTION_FRAMES = read_required<int>(*cfg, "MOTION_FRAMES");
        cfg_.MOTION_ANGLE_TOLERANCE_DEG = read_required<float>(*cfg, "MOTION_ANGLE_TOLERANCE_DEG");
        cfg_.OVERLAY_TTL_SECONDS = read_required<int>(*cfg, "OVERLAY_TTL_SECONDS");
        cfg_.TRACKER_INIT_PADDING = read_required<int>(*cfg, "TRACKER_INIT_PADDING");
        cfg_.TRACKER_MIN_SIZE = read_required<int>(*cfg, "TRACKER_MIN_SIZE");
        cfg_.MOTION_MIN_MAGNITUDE = read_required<float>(*cfg, "MOTION_MIN_MAGNITUDE");
        cfg_.MOTION_MAG_TOLERANCE_PX = read_required<float>(*cfg, "MOTION_MAG_TOLERANCE_PX");
        cfg_.MOTION_MAX_FEATURES = read_required<int>(*cfg, "MOTION_MAX_FEATURES");
        cfg_.MOTION_QUALITY_LEVEL = read_required<float>(*cfg, "MOTION_QUALITY_LEVEL");
        cfg_.MOTION_MIN_DISTANCE = read_required<float>(*cfg, "MOTION_MIN_DISTANCE");
        cfg_.MOTION_ANGLE_BIN_DEG = read_required<float>(*cfg, "MOTION_ANGLE_BIN_DEG");
        cfg_.MOTION_MAG_BIN_PX = read_required<float>(*cfg, "MOTION_MAG_BIN_PX");
        cfg_.MOTION_GRID_STEP_RATIO = read_required<float>(*cfg, "MOTION_GRID_STEP_RATIO");
        cfg_.MOTION_MIN_STABLE_RATIO = read_required<float>(*cfg, "MOTION_MIN_STABLE_RATIO");
        cfg_.WATCHDOG_PERIOD_MS = read_required<int>(*cfg, "WATCHDOG_PERIOD_MS");
        cfg_.WATCHDOG_MOTION_RATIO = read_required<float>(*cfg, "WATCHDOG_MOTION_RATIO");
        cfg_.WATCHDOG_ANGLE_TOLERANCE_DEG = read_required<float>(*cfg, "WATCHDOG_ANGLE_TOLERANCE_DEG");
        cfg_.WATCHDOG_FLOW_PYR_SCALE = read_required<double>(*cfg, "WatchdogFlowPyrScale");
        cfg_.WATCHDOG_FLOW_LEVELS = read_required<int>(*cfg, "WatchdogFlowLevels");
        cfg_.WATCHDOG_FLOW_WINSIZE = read_required<int>(*cfg, "WatchdogFlowWinSize");
        cfg_.WATCHDOG_FLOW_ITERATIONS = read_required<int>(*cfg, "WatchdogFlowIterations");
        cfg_.WATCHDOG_FLOW_POLY_N = read_required<int>(*cfg, "WatchdogFlowPolyN");
        cfg_.WATCHDOG_FLOW_POLY_SIGMA = read_required<double>(*cfg, "WatchdogFlowPolySigma");
        cfg_.WATCHDOG_FLOW_FLAGS = read_required<int>(*cfg, "WatchdogFlowFlags");
        cfg_.VISIBILITY_HISTORY_SIZE = read_required<int>(*cfg, "VISIBILITY_HISTORY_SIZE");
        cfg_.RESERVED_CANDIDATE_TTL_MS = read_required<int>(*cfg, "RESERVED_CANDIDATE_TTL_MS");
        cfg_.RESERVED_DETECTION_RADIUS_PX = read_required<float>(*cfg, "ReservedDetectionRadiusPx");
        cfg_.AUTO_DETECTION_PERIOD_MS = read_required<int>(*cfg, "AUTO_DETECTION_PERIOD_MS");
        cfg_.MOTION_DETECTION_ITERATIONS = read_required<int>(*cfg, "MOTION_DETECTION_ITERATIONS");
        cfg_.MOTION_DETECTION_DIFFUSION_PX = read_required<float>(*cfg, "MOTION_DETECTION_DIFFUSION_PX");
        cfg_.MOTION_DETECTION_CLUSTER_RATIO = read_required<float>(*cfg, "MOTION_DETECTION_CLUSTER_RATIO");
        cfg_.AUTO_HISTORY_SIZE = read_required<int>(*cfg, "AUTO_HISTORY_SIZE");
        cfg_.AUTO_DIFF_THRESHOLD = read_required<int>(*cfg, "AUTO_DIFF_THRESHOLD");
        cfg_.MOTION_BINARY_MAX_VALUE = read_required<int>(*cfg, "MotionBinaryMaxValue");
        cfg_.AUTO_MIN_AREA = read_required<double>(*cfg, "AUTO_MIN_AREA");
        cfg_.FLOODFILL_FILL_OVERLAY = read_required<bool>(*cfg, "FLOODFILL_FILL_OVERLAY");
        cfg_.FLOODFILL_OVERLAY_ALPHA = read_required<float>(*cfg, "FLOODFILL_OVERLAY_ALPHA");
        cfg_.MIN_AREA = read_required<int>(*cfg, "MIN_AREA");
        cfg_.MIN_WIDTH = read_required<int>(*cfg, "MIN_WIDTH");
        cfg_.MIN_HEIGHT = read_required<int>(*cfg, "MIN_HEIGHT");
        cfg_.TRACKER_TYPE = read_required<std::string>(*cfg, "TRACKER_TYPE");
        cfg_.VISIBILITY_HISTORY_SIZE = std::max(1, cfg_.VISIBILITY_HISTORY_SIZE);
        cfg_.RESERVED_CANDIDATE_TTL_MS = std::max(0, cfg_.RESERVED_CANDIDATE_TTL_MS);
        cfg_.AUTO_DETECTION_PERIOD_MS = std::max(1, cfg_.AUTO_DETECTION_PERIOD_MS);
        return true;
    } catch (const std::exception &e) {
        std::cerr << "[MANUAL] config load failed: " << e.what() << std::endl;
        return false;
    }
}

// Проверяет попадание клика в bbox с расширением по краям.
// Нужна для логики "клик по зелёному боксу удаляет цель".
bool ClickedTracksHandler::point_in_rect_with_padding(const cv::Rect2f &rect, int x, int y, int pad) const {
    cv::Rect2f padded(rect.x - pad, rect.y - pad, rect.width + pad * 2.0f, rect.height + pad * 2.0f);
    return padded.contains(cv::Point2f(static_cast<float>(x), static_cast<float>(y)));
}

// Записывает видимость трека в кольцевую историю последних N кадров.
// Это ядр правила "N кадров без обновления -> показать серый bbox".
void ClickedTracksHandler::record_visibility(ClickedTrack &track, bool visible) {
    const size_t history_size = track.visibility_history.size();
    if (history_size == 0) {
        return;
    }
    track.visibility_history[track.visibility_index] = visible;
    track.visibility_index = (track.visibility_index + 1) % history_size;
}

// Проверяет, что трек не был виден в последних N кадрах.
// Возвращает true, когда цель считается потерянной для отрисовки.
bool ClickedTracksHandler::has_recent_visibility_loss(const ClickedTrack &track) const {
    if (track.visibility_history.empty()) {
        return false;
    }
    return std::all_of(track.visibility_history.begin(),
                       track.visibility_history.end(),
                       [](bool visible) { return !visible; });
}


// Если синхронное движение не обнаружено — трек принудительно переводится в режим потери (серый bbox) и запускается поиск кандидата.
// Запускается по watchdog в update, чтобы убивать "залипшие" статические треки отвязавшиеся от целей.
// метод реализует детекцию синхронного движения:  cv::calcOpticalFlowFarneback(prev_roi, curr_roi, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
// Работает совместно с ClickedTracksHandler::mark_track_lost
//Логика:
//Оптический поток рассчитывается в ROI. Усредняется направление движения.
//Считается доля пикселей, чьи векторы движения совпадают по направлению с усреднённым.
//Если доля меньше заданного порога — считается, что движения нет.
bool ClickedTracksHandler::has_group_motion(const cv::Mat &prev_gray,
                                            const cv::Mat &curr_gray,
                                            const cv::Rect2f &roi) const {
    if (prev_gray.empty() || curr_gray.empty()) {
        return false;
    }
    cv::Rect clipped = to_int_rect(clip_rect(roi, curr_gray.size()));
    if (clipped.area() <= 0) {
        return false;
    }
    const cv::Mat prev_roi = prev_gray(clipped);
    const cv::Mat curr_roi = curr_gray(clipped);
    cv::Mat flow;
    cv::calcOpticalFlowFarneback(prev_roi,
                                 curr_roi,
                                 flow,
                                 cfg_.WATCHDOG_FLOW_PYR_SCALE,
                                 cfg_.WATCHDOG_FLOW_LEVELS,
                                 cfg_.WATCHDOG_FLOW_WINSIZE,
                                 cfg_.WATCHDOG_FLOW_ITERATIONS,
                                 cfg_.WATCHDOG_FLOW_POLY_N,
                                 cfg_.WATCHDOG_FLOW_POLY_SIGMA,
                                 cfg_.WATCHDOG_FLOW_FLAGS);

    const int total_pixels = clipped.width * clipped.height;
    if (total_pixels <= 0) {
        return false;
    }

    double sum_dx = 0.0;
    double sum_dy = 0.0;
    int moving_pixels = 0;
    for (int y = 0; y < flow.rows; ++y) {
        const auto *row = flow.ptr<cv::Vec2f>(y);
        for (int x = 0; x < flow.cols; ++x) {
            const cv::Vec2f vec = row[x];
            const float mag = std::hypot(vec[0], vec[1]);
            if (mag >= cfg_.MOTION_MIN_MAGNITUDE) {
                sum_dx += vec[0];
                sum_dy += vec[1];
                ++moving_pixels;
            }
        }
    }

    if (moving_pixels == 0) {
        return false;
    }

    const double mean_dx = sum_dx / moving_pixels;
    const double mean_dy = sum_dy / moving_pixels;
    const double mean_mag = std::hypot(mean_dx, mean_dy);
    if (mean_mag < cfg_.MOTION_MIN_MAGNITUDE) {
        return false;
    }

    const double cos_tol = std::cos(cfg_.WATCHDOG_ANGLE_TOLERANCE_DEG * CV_PI / 180.0);
    int aligned_pixels = 0;
    for (int y = 0; y < flow.rows; ++y) {
        const auto *row = flow.ptr<cv::Vec2f>(y);
        for (int x = 0; x < flow.cols; ++x) {
            const cv::Vec2f vec = row[x];
            const float mag = std::hypot(vec[0], vec[1]);
            if (mag < cfg_.MOTION_MIN_MAGNITUDE) {
                continue;
            }
            const double dot = vec[0] * mean_dx + vec[1] * mean_dy;
            const double cos_angle = dot / (mag * mean_mag);
            if (cos_angle >= cos_tol) {
                ++aligned_pixels;
            }
        }
    }

    return static_cast<float>(aligned_pixels) >= static_cast<float>(total_pixels) * cfg_.WATCHDOG_MOTION_RATIO;
}


// Метод перевода трека в состояние потери цели. Работает совместно с has_group_motion
void ClickedTracksHandler::mark_track_lost(ClickedTrack &track, long long now_ms) {
    track.lost_since_ms = now_ms;
    if (track.visibility_history.empty()) {
        track.visibility_history.assign(static_cast<size_t>(cfg_.VISIBILITY_HISTORY_SIZE), false);
    } else {
        std::fill(track.visibility_history.begin(), track.visibility_history.end(), false);
    }
    track.visibility_index = 0;
    track.candidate_search.reset();
    track.candidate_search.configure_motion_filter(
            cfg_.MOTION_DETECTION_ITERATIONS,
            cfg_.MOTION_DETECTION_DIFFUSION_PX,
            cfg_.MOTION_DETECTION_CLUSTER_RATIO,
            cfg_.AUTO_HISTORY_SIZE,
            cfg_.AUTO_DIFF_THRESHOLD,
            cfg_.AUTO_MIN_AREA
    );
    track.candidate_search.set_reserved_detection_radius(cfg_.RESERVED_DETECTION_RADIUS_PX);
}

// Создаёт экземпляр OpenCV-трекера на основе настроек.
// Поддерживает KCF и CSRT.
cv::Ptr <cv::Tracker> ClickedTracksHandler::create_tracker() const {
    std::string type = cfg_.TRACKER_TYPE;
    std::transform(type.begin(), type.end(), type.begin(), [](unsigned char ch) {
        return static_cast<char>(std::toupper(ch));
    });
    if (type == "CSRT") {
        return cv::TrackerCSRT::create();
    }
    return cv::TrackerKCF::create();
}

// Обработчик ЛКМ: удаляет цель по клику по bbox или инициирует новую.
// Клик по пустому месту создаёт PendingClick для анализа движения.
bool ClickedTracksHandler::handle_click(int x, int y, const cv::Mat &frame, long long now_ms) {
    std::lock_guard <std::mutex> lock(mutex_);
    (void) now_ms;

    // Сначала проверяем, не кликнули ли по существующему боксу — тогда удаляем цель.
    // Цикл: ищет первый трек, чей bbox покрывает клик (с паддингом), и удаляет его.
    for (auto it = tracks_.begin(); it != tracks_.end(); ++it) {
        if (point_in_rect_with_padding(it->bbox, x, y, cfg_.CLICK_PADDING)) {
            tracks_.erase(it);
            refresh_targets();
            return true;
        }
    }

    if (static_cast<int>(tracks_.size()) >= cfg_.MAX_TARGETS) {
        return false;
    }

    if (frame.empty()) {
        return false;
    }

    cv::Rect roi = clicked_target_shaper_.make_click_roi(frame, x, y);
    if (roi.area() <= 0) {
        return false;
    }

    // Клик → подозрение на объект. Несколько кадров подряд анализируется движение в ROI.
    // Если движение подтверждено, создаётся трек и запускается OpenCV-трекер.
    PendingClick pending;
    pending.roi = roi;
    pending.gray_frames.push_back(to_gray(frame));
    pending_clicks_.push_back(std::move(pending));
    if (log_cfg_.MANUAL_DETECTOR_LEVEL_LOGGER) {
        std::cout << "[MANUAL] pending motion click id=" << next_id_
                  << " roi=" << roi.width << "x" << roi.height << std::endl;
    }
    return true;
}

// Основной цикл обновления: обрабатывает pending-клики, треки и выводит Target-ы.
// Здесь же формируется логика "N кадров без обновления -> серый bbox".
void ClickedTracksHandler::update(cv::Mat &frame, long long now_ms) {
    std::lock_guard <std::mutex> lock(mutex_);

    if (frame.empty()) {
        refresh_targets();
        return;
    }

    // Цепочка 2: фоновые детекции обновляются каждый кадр и используют периодизацию.
    motion_detector_.update(frame, now_ms);
    cv::Mat current_gray = to_gray(frame);
    const bool should_run_watchdog = !watchdog_prev_gray_.empty()
                                     && (now_ms - watchdog_prev_ms_ >= cfg_.WATCHDOG_PERIOD_MS);

    std::vector <cv::Rect2f> tracked_boxes; // bbox видимых треков; используются как список запрета для выбора кандидатов.
    tracked_boxes.reserve(
            tracks_.size()); // резервируем место под bbox всех видимых треков, чтобы избежать реаллокаций.
    // Цикл: собирает bbox видимых треков для фильтрации кандидатов автопоиска.
    for (const auto &track: tracks_) {
        if (track.lost_since_ms == 0) {
            tracked_boxes.push_back(track.bbox); //tracked_boxes - зеленые ббоксы
        }
    }


// Этот фрагмент — классическая erase–remove идиома для удаления элементов из std::vector по условию.
    reserved_candidates_.erase(
            // remove_if -проходит по reserved_candidates_ и перемещает все элементы, для которых условие лямбды ложно, в начало диапазона
            //Лямбда возвращает true, когда кандидат просрочен (candidate.expires_ms <= now_ms).
            //Эти «плохие» элементы сдвигаются в конец, но размер вектора не меняется.
            //reserved_candidates_.erase(new_end, reserved_candidates_.end())  - фактически удаляет хвостовой диапазон «просроченных» элементов, который пометил remove_if.
            std::remove_if(reserved_candidates_.begin(), reserved_candidates_.end(),
                           [&](const ReservedCandidate &candidate) {
                               return candidate.expires_ms <= now_ms;
                           }),
            reserved_candidates_.end());
// после выполнения в reserved_candidates_ остаются только кандидаты, у которых expires_ms > now_ms (то есть ещё «живые»).


    if (!pending_clicks_.empty()) {
        cv::Mat gray = current_gray;
        // Цикл: дополняет историю кадров для pending-кликов и, при готовности,
        //       пытается создать трек на основе движения в ROI.
        // Цепочка 1: клик → накопление кадров → clicked_target_shaper_ → bbox для трекера.
        for (auto it = pending_clicks_.begin(); it != pending_clicks_.end();) {
            it->gray_frames.push_back(gray);
            // Цепочка 1: подтверждение клика оператором и формирование bbox.
            const int required = clicked_target_shaper_.required_frames();
            if (static_cast<int>(it->gray_frames.size()) < required) {
                ++it;
                continue;
            }

            if (static_cast<int>(tracks_.size()) >= cfg_.MAX_TARGETS) {
                it = pending_clicks_.erase(it);
                continue;
            }

            std::vector <cv::Point2f> motion_points;
            cv::Rect2f motion_roi;
            cv::Rect2f tracker_roi;
            if (clicked_target_shaper_.build_candidate(it->gray_frames, it->roi, frame.size(),
                                                       tracker_roi, &motion_points, &motion_roi)) {
                if (log_cfg_.MANUAL_DETECTOR_LEVEL_LOGGER) {
                    std::cout << "[MANUAL] capture dynamic click id=" << next_id_ << std::endl;
                }

                ClickedTrack track;
                track.id = next_id_++;
                track.bbox = tracker_roi;
                track.tracker = create_tracker();
                track.tracker->init(frame, track.bbox);
                track.lost_since_ms = 0;
                track.visibility_history.assign(static_cast<size_t>(cfg_.VISIBILITY_HISTORY_SIZE), true);
                track.visibility_index = 0;
                track.cross_center = (motion_roi.area() > 1.0f)
                                     ? rect_center(motion_roi)
                                     : rect_center(track.bbox);
                track.last_known_center = track.cross_center;
                track.candidate_search.configure(&clicked_target_shaper_, &motion_detector_);
                track.candidate_search.configure_motion_filter(
                        cfg_.MOTION_DETECTION_ITERATIONS,
                        cfg_.MOTION_DETECTION_DIFFUSION_PX,
                        cfg_.MOTION_DETECTION_CLUSTER_RATIO,
                        cfg_.AUTO_HISTORY_SIZE,
                        cfg_.AUTO_DIFF_THRESHOLD,
                        cfg_.AUTO_MIN_AREA
                );
                track.candidate_search.set_reserved_detection_radius(cfg_.RESERVED_DETECTION_RADIUS_PX);

                tracks_.push_back(std::move(track));
                refresh_targets();

                if (cfg_.FLOODFILL_FILL_OVERLAY) {
                    flood_fill_mask_ = cv::Mat::zeros(frame.size(), CV_8UC1);
                    flood_fill_overlay_ = cv::Mat::zeros(frame.size(), frame.type());
                    if (motion_points.size() >= 3) {
                        std::vector <cv::Point2f> hull;
                        cv::convexHull(motion_points, hull, true);
                        std::vector <cv::Point> hull_int;
                        hull_int.reserve(hull.size());
                        // Цикл: переводит точки hull в целочисленные координаты маски заливк.
                        for (const auto &pt: hull) {
                            hull_int.emplace_back(static_cast<int>(std::round(pt.x)),
                                                  static_cast<int>(std::round(pt.y)));
                        }
                        if (hull_int.size() >= 3) {
                            cv::fillConvexPoly(flood_fill_mask_, hull_int, cv::Scalar(255), cv::LINE_AA);
                        }
                    } else {
                        cv::Rect overlay_rect = to_int_rect(motion_roi);
                        overlay_rect &= cv::Rect(0, 0, frame.cols, frame.rows);
                        if (overlay_rect.area() > 0) {
                            flood_fill_mask_(overlay_rect).setTo(255);
                        }
                    }
                    flood_fill_overlay_.setTo(cv::Scalar(0, 0, 255), flood_fill_mask_);
                    overlay_expire_ms_ = now_ms + static_cast<long long>(cfg_.OVERLAY_TTL_SECONDS) * 1000;
                }

                if (log_cfg_.TRACKER_LEVEL_LOGGER) {
                    std::cout << "[TRK] start dynamic id=" << (next_id_ - 1) << std::endl;
                }
                if (log_cfg_.TARGET_OBJECT_CREATED_LOGGER) {
                    std::cout << "[MANUAL] target object created id=" << (next_id_ - 1) << std::endl;
                }
            } else if (log_cfg_.MANUAL_DETECTOR_LEVEL_LOGGER) {
                std::cout << "[MANUAL] motion click ignored (static) id=" << next_id_ << std::endl;
            }
            it = pending_clicks_.erase(it);
        }
    }

    // Цикл: обновляет все активные треки (OpenCV-трекер, потери, автопоиск кандидатов).
    for (auto it = tracks_.begin(); it != tracks_.end();) {
// tracked_boxes — список bbox видимых треков (не потерянных) (lost_since_ms == 0) треков.
// Этот список нужен, чтобы при поиске кандидата исключать точки, уже находящиеся внутри активных треков.
// эти bbox используются для фильтрации кандидатов и не должны попадать в новый поиск.
        std::vector <cv::Rect2f> reserved_boxes = tracked_boxes;
// предварительное резервирование памяти под bbox видимых треков, чтобы избежать лишних реаллокаций при push_back.
        reserved_boxes.reserve(tracked_boxes.size() + reserved_candidates_.size());


// в этом цикле для каждого трека it, в reserved_candidates_ добавляются все незаэкспайреные кандидаты недавно выданные текущему треку
        for (const auto &candidate: reserved_candidates_) {
            if (candidate.owner_id == it->id) {
                continue;
            }
            reserved_boxes.push_back(candidate.bbox);
        }

/* reserved_candidates_ — вектор структур {bbox, expires_ms, owner_id}, то есть кандидаты, закреплённые за конкретными треками на TTL. Используется как механизм блокировки, чтобы один и тот же кандидат не отдавался другим трекам одновременно.
   Для каждого трека строятся локальные reserved_boxes и reserved_points, и они передаются в AutoCandidateSearch.

*/      it->candidate_search.set_tracked_boxes(reserved_boxes);
        std::vector <cv::Point2f> reserved_points; // центры зарезервированных кандидатов (запрещены в радиусе reserved_detection_radius_).
        reserved_points.reserve(
                reserved_candidates_.size()); // резервируем место под центры всех зарезервированных кандидатов.
        for (const auto &candidate: reserved_candidates_) {
            if (candidate.owner_id == it->id) {
                continue;
            }
            reserved_points.push_back(
                    rect_center(candidate.bbox)); // фиксируем центр кандидата для запрета соседних детекций.
        }
/* В эти списки включаются bbox/центры резервов других треков, а собственный резерв конкретного трека исключается.
Так достигается блокировка кандидатов для всех остальных треков.
*/        it->candidate_search.set_reserved_detection_points(reserved_points);

        bool visible = false;
        const cv::Rect2f prev_bbox = it->bbox;
        if (it->lost_since_ms == 0 && !frame.empty() && it->tracker) {
            cv::Rect new_box;
            if (it->tracker->update(frame, new_box)) {
                const cv::Rect2f candidate = ::clip_rect(cv::Rect2f(new_box), frame.size());
                if (candidate.area() > 1.0f) {
                    it->bbox = candidate;
                } else {
                    it->bbox = prev_bbox;
                }
                if (it->bbox.area() > 1.0f) {
                    it->lost_since_ms = 0;
                    visible = true;
                    it->cross_center = rect_center(it->bbox);
                    it->last_known_center = it->cross_center;
                    it->candidate_search.reset();
                }
            }
        }

        record_visibility(*it, visible);
        if (!visible && it->lost_since_ms == 0) {
            mark_track_lost(*it, now_ms);
        }

        if (should_run_watchdog && it->lost_since_ms == 0) {
            if (!has_group_motion(watchdog_prev_gray_, current_gray, it->bbox)) {
                if (log_cfg_.MANUAL_DETECTOR_LEVEL_LOGGER) {
                    std::cout << "[MANUAL] watchdog motion lost id=" << it->id << std::endl;
                }
                mark_track_lost(*it, now_ms);
            }
        }

// ЗАПУСК ПОИСКА КАНДИДАТА на перезахват
        // вот блок запуска поиска кандидатов при потере треком цели (трек посерел)
        if (it->lost_since_ms > 0) {
            // Цепочка 2: потеря → AutoCandidateSearch → DetectionMatcher → пул MotionDetector.
            if (!it->candidate_search.active()) {
                it->candidate_search.start(it->last_known_center, now_ms, frame);
            }
            cv::Rect2f candidate_bbox;
            if (it->candidate_search.update(frame, candidate_bbox)) {
                if (log_cfg_.MANUAL_DETECTOR_LEVEL_LOGGER) {
                    std::cout << "[MANUAL] auto candidate acquired id=" << it->id << std::endl;
                }
                it->bbox = candidate_bbox;
                reserved_candidates_.push_back({candidate_bbox,
                                                now_ms + cfg_.RESERVED_CANDIDATE_TTL_MS,
                                                it->id});
                it->cross_center = rect_center(it->bbox);
                it->tracker = create_tracker();
                it->tracker->init(frame, it->bbox);
                it->lost_since_ms = 0;
                if (it->visibility_history.empty()) {
                    it->visibility_history.assign(static_cast<size_t>(cfg_.VISIBILITY_HISTORY_SIZE), true);
                } else {
                    std::fill(it->visibility_history.begin(), it->visibility_history.end(), true);
                }
                it->visibility_index = 0;
                it->last_known_center = it->cross_center;
                it->candidate_search.reset();
            }
        }
        ++it;
    }

    if (watchdog_prev_gray_.empty() || should_run_watchdog) {
        watchdog_prev_gray_ = current_gray.clone();
        watchdog_prev_ms_ = now_ms;
    }

    refresh_targets();

    if (overlay_expire_ms_ > 0 && now_ms >= overlay_expire_ms_) {
        flood_fill_overlay_.release();
        flood_fill_mask_.release();
        overlay_expire_ms_ = 0;
    }
    if (cfg_.FLOODFILL_FILL_OVERLAY && !flood_fill_overlay_.empty() && !flood_fill_mask_.empty()) {
        cv::Mat blended;
        const double overlay_alpha = cfg_.FLOODFILL_OVERLAY_ALPHA;
        cv::addWeighted(frame, 1.0 - overlay_alpha, flood_fill_overlay_, overlay_alpha, 0.0, blended);
        blended.copyTo(frame, flood_fill_mask_);
    }
}

// Преобразует текущие треки в выходной список целей.
// missed_frames используется только как сигнал "серый bbox" для рендера.
void ClickedTracksHandler::refresh_targets() {
    targets_.clear();
    targets_.reserve(tracks_.size());
    // Цикл: преобразует внутренние треки в Target для отрисовки/экспорта.
    for (const auto &tr: tracks_) {
        Target tg;
        tg.id = tr.id;
        tg.target_name = "T" + std::to_string(tr.id);
        tg.bbox = tr.bbox;
        tg.has_cross = true;
        tg.cross_center = tr.cross_center;
        // Переводим цель в "потерянную" после трёх последовательных промахов трекера.
        tg.missed_frames = tr.lost_since_ms > 0 ? 1 : (has_recent_visibility_loss(tr) ? 1 : 0);
        targets_.push_back(std::move(tg));
    }
}
