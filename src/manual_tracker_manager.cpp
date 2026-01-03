#include "manual_tracker_manager.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <string>

namespace {
    // Возвращает центр прямоугольника.
    // Используется для запоминания последней позиции цели.
    static inline cv::Point2f rect_center(const cv::Rect2f& rect) {
        return {rect.x + rect.width * 0.5f, rect.y + rect.height * 0.5f};
    }

    // Переводит прямоугольник с float координатами в int, округляя границы.
    // Нужен для отрисовки и операций с матрицами масок.
    static inline cv::Rect to_int_rect(const cv::Rect2f& rect) {
        return cv::Rect(
                static_cast<int>(std::round(rect.x)),
                static_cast<int>(std::round(rect.y)),
                static_cast<int>(std::round(rect.width)),
                static_cast<int>(std::round(rect.height))
        );
    }

    // Переводит кадр в градации серого, если он в формате BGR.
    // Клонирует кадр, чтобы не модифицировать исходные данные.
    static inline cv::Mat to_gray(const cv::Mat& frame) {
        if (frame.channels() == 3) {
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            return gray;
        }
        return frame.clone();
    }

    // Обрезает прямоугольник по размеру кадра, исключая отрицательные координаты.
    // Гарантирует корректный bbox перед передчей трекеру или рендереру.
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

    // Размер кольцевого буфера видимости: три кадра для определения потери цели.
    constexpr size_t kVisibilityHistorySize = 3;
}

// Конструктор: поднимает логирование и загружает настройки ручного трекера.
ManualTrackerManager::ManualTrackerManager(const toml::table& tbl) {
    load_logging_config(tbl, log_cfg_);
    load_config(tbl);
    ManualMotionDetectorConfig motion_cfg;
    motion_cfg.click_capture_size = cfg_.click_capture_size;
    motion_cfg.motion_frames = cfg_.motion_frames;
    motion_cfg.motion_diff_threshold = cfg_.motion_diff_threshold;
    motion_cfg.click_padding = cfg_.click_padding;
    motion_cfg.tracker_init_padding = cfg_.tracker_init_padding;
    motion_cfg.tracker_min_size = cfg_.tracker_min_size;
    motion_cfg.motion_min_magnitude = cfg_.motion_min_magnitude;
    motion_cfg.motion_mag_tolerance_px = cfg_.motion_mag_tolerance_px;
    motion_cfg.min_area = cfg_.min_area;
    motion_cfg.min_width = cfg_.min_width;
    motion_cfg.min_height = cfg_.min_height;
    motion_detector_.update_config(motion_cfg);
}

// Загружает параметры ручного трекера из TOML и печатает краткую сводку.
// Любая ошибка в таблице [manual_tracker] приводит к исключению и сообщению в лог.
bool ManualTrackerManager::load_config(const toml::table& tbl) {
    try {
        const auto *cfg = tbl["manual_tracker"].as_table();
        if (!cfg) {
            throw std::runtime_error("missing [manual_tracker] table");
        }
        cfg_.max_targets = read_required<int>(*cfg, "max_targets");
        cfg_.click_padding = read_required<int>(*cfg, "click_padding");
        cfg_.motion_diff_threshold = read_required<int>(*cfg, "motion_diff_threshold");
        cfg_.click_capture_size = read_required<int>(*cfg, "click_capture_size");
        cfg_.motion_frames = read_required<int>(*cfg, "motion_frames");
        cfg_.overlay_ttl_seconds = read_required<int>(*cfg, "overlay_ttl_seconds");
        cfg_.tracker_init_padding = read_required<int>(*cfg, "tracker_init_padding");
        cfg_.tracker_min_size = read_required<int>(*cfg, "tracker_min_size");
        cfg_.motion_min_magnitude = read_required<float>(*cfg, "motion_min_magnitude");
        cfg_.motion_mag_tolerance_px = read_required<float>(*cfg, "motion_mag_tolerance_px");
        cfg_.tracker_rebind_ms = read_required<int>(*cfg, "tracker_rebind_ms");
        cfg_.floodfill_fill_overlay = read_required<bool>(*cfg, "floodfill_fill_overlay");
        cfg_.floodfill_lo_diff = read_required<int>(*cfg, "floodfill_lo_diff");
        cfg_.floodfill_hi_diff = read_required<int>(*cfg, "floodfill_hi_diff");
        cfg_.min_area = read_required<int>(*cfg, "min_area");
        cfg_.min_width = read_required<int>(*cfg, "min_width");
        cfg_.min_height = read_required<int>(*cfg, "min_height");
        cfg_.tracker_type = read_required<std::string>(*cfg, "tracker_type");
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[MANUAL] config load failed: " << e.what() << std::endl;
        return false;
    }
}

// Проверяет попадание клика в bbox с расширением по краям.
// Нужна для логики "клик по зелёному боксу удаляет цель".
bool ManualTrackerManager::point_in_rect_with_padding(const cv::Rect2f& rect, int x, int y, int pad) const {
    cv::Rect2f padded(rect.x - pad, rect.y - pad, rect.width + pad * 2.0f, rect.height + pad * 2.0f);
    return padded.contains(cv::Point2f(static_cast<float>(x), static_cast<float>(y)));
}

// Записывает видимость трека в кольцевую историю последних трех кадров.
// Это ядро правила "3 кадра без обновления -> показать серый bbox".
void ManualTrackerManager::record_visibility(ManualTrack& track, bool visible) {
    track.visibility_history[track.visibility_index] = visible;
    track.visibility_index = (track.visibility_index + 1) % kVisibilityHistorySize;
}

// Проверяет, что трек не был виден в последних трех кадрах.
// Возвращает true, когда цель считается потерянной для отрисовки.
bool ManualTrackerManager::has_recent_visibility_loss(const ManualTrack& track) const {
    return std::all_of(track.visibility_history.begin(),
                       track.visibility_history.end(),
                       [](bool visible) { return !visible; });
}

// Создаёт экземпляр OpenCV-трекера на основе настроек.
// Поддерживает KCF и CSRT.
cv::Ptr<cv::Tracker> ManualTrackerManager::create_tracker() const {
    std::string type = cfg_.tracker_type;
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
bool ManualTrackerManager::handle_click(int x, int y, const cv::Mat& frame, long long now_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    (void)now_ms;

    // Сначала проверяем, не кликнули ли по существующему боксу — тогда удаляем цель.
    for (auto it = tracks_.begin(); it != tracks_.end(); ++it) {
        if (point_in_rect_with_padding(it->bbox, x, y, cfg_.click_padding)) {
            tracks_.erase(it);
            refresh_targets();
            return true;
        }
    }

    if (static_cast<int>(tracks_.size()) >= cfg_.max_targets) {
        return false;
    }

    if (frame.empty()) {
        return false;
    }

    cv::Rect roi = motion_detector_.make_click_roi(frame, x, y);
    if (roi.area() <= 0) {
        return false;
    }

    // Клик → подозрение на объект. Несколько кадров подряд анализируется движение в ROI.
    // Если движение подтверждено, создаётся трек и запускается OpenCV-трекер.
    PendingClick pending;
    pending.roi = roi;
    pending.gray_frames.push_back(to_gray(frame));
    pending_clicks_.push_back(std::move(pending));
    if (log_cfg_.manual_detector_level_logger) {
        std::cout << "[MANUAL] pending motion click id=" << next_id_
                  << " roi=" << roi.width << "x" << roi.height << std::endl;
    }
    return true;
}

// Основной цикл обновления: обрабатывает pending-клики, треки и выводит Target-ы.
// Здесь же формируется логика "3 кадра без обновления -> серый bbox".
void ManualTrackerManager::update(cv::Mat& frame, long long now_ms) {
    std::lock_guard<std::mutex> lock(mutex_);

    cv::Mat current_gray;
    if (!frame.empty()) {
        current_gray = to_gray(frame);
    }

    std::vector<cv::Rect2f> tracked_boxes;
    tracked_boxes.reserve(tracks_.size());
    for (const auto& track : tracks_) {
        tracked_boxes.push_back(track.bbox);
    }

    if (!pending_clicks_.empty() && !frame.empty()) {
        cv::Mat gray = current_gray;
        for (auto it = pending_clicks_.begin(); it != pending_clicks_.end(); ) {
            it->gray_frames.push_back(gray);
            const int required = motion_detector_.required_frames();
            if (static_cast<int>(it->gray_frames.size()) < required) {
                ++it;
                continue;
            }

            if (static_cast<int>(tracks_.size()) >= cfg_.max_targets) {
                it = pending_clicks_.erase(it);
                continue;
            }

            std::vector<cv::Point2f> motion_points;
            cv::Rect2f motion_roi;
            cv::Rect2f tracker_roi;
            if (motion_detector_.build_candidate(it->gray_frames, it->roi, frame.size(),
                                                 tracker_roi, &motion_points, &motion_roi)) {
                if (log_cfg_.manual_detector_level_logger) {
                    std::cout << "[MANUAL] capture dynamic click id=" << next_id_ << std::endl;
                }

                ManualTrack track;
                track.id = next_id_++;
                track.bbox = tracker_roi;
                track.tracker = create_tracker();
                track.tracker->init(frame, track.bbox);
                track.lost_since_ms = 0;
                track.visibility_history.fill(true);
                track.visibility_index = 0;
                track.last_known_center = rect_center(track.bbox);
                track.candidate_search.configure(&motion_detector_);

                tracks_.push_back(std::move(track));
                refresh_targets();

                if (cfg_.floodfill_fill_overlay) {
                    flood_fill_mask_ = cv::Mat::zeros(frame.size(), CV_8UC1);
                    flood_fill_overlay_ = cv::Mat::zeros(frame.size(), frame.type());
                    if (motion_points.size() >= 3) {
                        std::vector<cv::Point2f> hull;
                        cv::convexHull(motion_points, hull, true);
                        std::vector<cv::Point> hull_int;
                        hull_int.reserve(hull.size());
                        for (const auto& pt : hull) {
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
                    overlay_expire_ms_ = now_ms + static_cast<long long>(cfg_.overlay_ttl_seconds) * 1000;
                }

                if (log_cfg_.tracker_level_logger) {
                    std::cout << "[TRK] start dynamic id=" << (next_id_ - 1) << std::endl;
                }
                if (log_cfg_.target_object_created_logger) {
                    std::cout << "[MANUAL] target object created id=" << (next_id_ - 1) << std::endl;
                }
            } else if (log_cfg_.manual_detector_level_logger) {
                std::cout << "[MANUAL] motion click ignored (static) id=" << next_id_ << std::endl;
            }
            it = pending_clicks_.erase(it);
        }
    }

    for (auto it = tracks_.begin(); it != tracks_.end(); ) {
        it->candidate_search.set_tracked_boxes(tracked_boxes);
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
                    it->last_known_center = rect_center(it->bbox);
                    it->candidate_search.reset();
                }
            }
        }

        if (!frame.empty()) {
            record_visibility(*it, visible);
            if (!visible && it->lost_since_ms == 0) {
                it->lost_since_ms = now_ms;
                it->visibility_history.fill(false);
                it->visibility_index = 0;
                it->candidate_search.reset();
            }
        }

        if (it->lost_since_ms > 0) {
            if (!frame.empty()) {
                const long long lost_for_ms = now_ms - it->lost_since_ms;
                if (lost_for_ms >= cfg_.tracker_rebind_ms) {
                    if (!it->candidate_search.active()) {
                        it->candidate_search.start(it->last_known_center, now_ms, frame);
                    }
                    cv::Rect2f candidate_bbox;
                    if (it->candidate_search.update(frame, candidate_bbox)) {
                        if (log_cfg_.manual_detector_level_logger) {
                            std::cout << "[MANUAL] auto candidate acquired id=" << it->id << std::endl;
                        }
                        it->bbox = candidate_bbox;
                        it->tracker = create_tracker();
                        it->tracker->init(frame, it->bbox);
                        it->lost_since_ms = 0;
                        it->visibility_history.fill(true);
                        it->visibility_index = 0;
                        it->last_known_center = rect_center(it->bbox);
                        it->candidate_search.reset();
                    }
                }
            }
        }
        ++it;
    }

    refresh_targets();

    if (frame.empty()) {
        return;
    }
    if (overlay_expire_ms_ > 0 && now_ms >= overlay_expire_ms_) {
        flood_fill_overlay_.release();
        flood_fill_mask_.release();
        overlay_expire_ms_ = 0;
    }
    if (cfg_.floodfill_fill_overlay && !flood_fill_overlay_.empty() && !flood_fill_mask_.empty()) {
        cv::Mat blended;
        constexpr double kOverlayAlpha = 0.7;
        cv::addWeighted(frame, 1.0 - kOverlayAlpha, flood_fill_overlay_, kOverlayAlpha, 0.0, blended);
        blended.copyTo(frame, flood_fill_mask_);
    }
}

// Преобразует текущие треки в выходной список целей.
// missed_frames используется только как сигнал "серый bbox" для рендера.
void ManualTrackerManager::refresh_targets() {
    targets_.clear();
    targets_.reserve(tracks_.size());
    for (const auto& tr : tracks_) {
        Target tg;
        tg.id = tr.id;
        tg.target_name = "T" + std::to_string(tr.id);
        tg.bbox = tr.bbox;
        // Переводим цель в "потерянную" после трёх последовательных промахов трекера.
        tg.missed_frames = tr.lost_since_ms > 0 ? 1 : (has_recent_visibility_loss(tr) ? 1 : 0);
        targets_.push_back(std::move(tg));
    }
}
