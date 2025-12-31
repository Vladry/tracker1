#include "manual_tracker_manager.h"
#include <algorithm>
#include <cmath>
#include <cctype>
#include <iostream>
#include <limits>
#include <string>

namespace {
    static inline cv::Point2f rect_center(const cv::Rect2f& rect) {
        return {rect.x + rect.width * 0.5f, rect.y + rect.height * 0.5f};
    }

    static inline cv::Rect to_int_rect(const cv::Rect2f& rect) {
        return cv::Rect(
                static_cast<int>(std::round(rect.x)),
                static_cast<int>(std::round(rect.y)),
                static_cast<int>(std::round(rect.width)),
                static_cast<int>(std::round(rect.height))
        );
    }
}

ManualTrackerManager::ManualTrackerManager(const toml::table& tbl) {
    load_logging_config(tbl, log_cfg_);
    load_config(tbl);
}

bool ManualTrackerManager::load_config(const toml::table& tbl) {
    try {
        const auto *cfg = tbl["manual_tracker"].as_table();
        if (!cfg) {
            throw std::runtime_error("missing [manual_tracker] table");
        }
        cfg_.max_targets = read_required<int>(*cfg, "max_targets");
        cfg_.click_padding = read_required<int>(*cfg, "click_padding");
        cfg_.remove_padding = read_required<int>(*cfg, "remove_padding");
        cfg_.fallback_box_size = read_required<int>(*cfg, "fallback_box_size");
        cfg_.max_area_ratio = read_required<float>(*cfg, "max_area_ratio");
        cfg_.motion_search_radius = read_required<int>(*cfg, "motion_search_radius");
        cfg_.motion_diff_threshold = read_required<int>(*cfg, "motion_diff_threshold");
        cfg_.click_search_radius = read_required<int>(*cfg, "click_search_radius");
        cfg_.click_equalize = read_required<bool>(*cfg, "click_equalize");
        cfg_.floodfill_fill_overlay = read_required<bool>(*cfg, "floodfill_fill_overlay");
        cfg_.floodfill_lo_diff = read_required<int>(*cfg, "floodfill_lo_diff");
        cfg_.floodfill_hi_diff = read_required<int>(*cfg, "floodfill_hi_diff");
        cfg_.min_area = read_required<int>(*cfg, "min_area");
        cfg_.min_width = read_required<int>(*cfg, "min_width");
        cfg_.min_height = read_required<int>(*cfg, "min_height");
        cfg_.search_radius = read_required<int>(*cfg, "search_radius");
        cfg_.match_threshold = read_required<float>(*cfg, "match_threshold");
        cfg_.update_template = read_required<bool>(*cfg, "update_template");
        cfg_.max_lost_ms = read_required<int>(*cfg, "max_lost_ms");
        cfg_.auto_reacquire_nearest = read_required<bool>(*cfg, "auto_reacquire_nearest");
        cfg_.reacquire_delay_ms = read_required<int>(*cfg, "reacquire_delay_ms");
        cfg_.reacquire_max_distance_px = read_required<int>(*cfg, "reacquire_max_distance_px");
        cfg_.tracker_type = read_required<std::string>(*cfg, "tracker_type");
        cfg_.kalman_process_noise = read_required<float>(*cfg, "kalman_process_noise");
        cfg_.kalman_measurement_noise = read_required<float>(*cfg, "kalman_measurement_noise");
        std::cout << "[MANUAL] config: max_targets=" << cfg_.max_targets
                  << " click_padding=" << cfg_.click_padding
                  << " remove_padding=" << cfg_.remove_padding
                  << " fallback_box_size=" << cfg_.fallback_box_size
                  << " max_area_ratio=" << cfg_.max_area_ratio
                  << " motion_search_radius=" << cfg_.motion_search_radius
                  << " motion_diff_threshold=" << cfg_.motion_diff_threshold
                  << " click_search_radius=" << cfg_.click_search_radius
                  << " click_equalize=" << (cfg_.click_equalize ? "true" : "false")
                  << " floodfill_fill_overlay=" << (cfg_.floodfill_fill_overlay ? "true" : "false")
                  << " floodfill_lo_diff=" << cfg_.floodfill_lo_diff
                  << " floodfill_hi_diff=" << cfg_.floodfill_hi_diff
                  << " min_area=" << cfg_.min_area
                  << " min_width=" << cfg_.min_width
                  << " min_height=" << cfg_.min_height
                  << " search_radius=" << cfg_.search_radius
                  << " match_threshold=" << cfg_.match_threshold
                  << " update_template=" << (cfg_.update_template ? "true" : "false")
                  << " max_lost_ms=" << cfg_.max_lost_ms
                  << " auto_reacquire_nearest=" << (cfg_.auto_reacquire_nearest ? "true" : "false")
                  << " reacquire_delay_ms=" << cfg_.reacquire_delay_ms
                  << " reacquire_max_distance_px=" << cfg_.reacquire_max_distance_px
                  << " tracker_type=" << cfg_.tracker_type
                  << " kalman_process_noise=" << cfg_.kalman_process_noise
                  << " kalman_measurement_noise=" << cfg_.kalman_measurement_noise
                  << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[MANUAL] config load failed: " << e.what() << std::endl;
        return false;
    }
}

cv::Rect2f ManualTrackerManager::clip_rect(const cv::Rect2f& rect, const cv::Size& size) const {
    float x1 = std::max(0.0f, rect.x);
    float y1 = std::max(0.0f, rect.y);
    float x2 = std::min(rect.x + rect.width, static_cast<float>(size.width));
    float y2 = std::min(rect.y + rect.height, static_cast<float>(size.height));
    if (x2 <= x1 || y2 <= y1) {
        return {};
    }
    return {x1, y1, x2 - x1, y2 - y1};
}

bool ManualTrackerManager::point_in_rect_with_padding(const cv::Rect2f& rect, int x, int y, int pad) const {
    cv::Rect2f padded(rect.x - pad, rect.y - pad, rect.width + pad * 2.0f, rect.height + pad * 2.0f);
    return padded.contains(cv::Point2f(static_cast<float>(x), static_cast<float>(y)));
}

cv::Rect2f ManualTrackerManager::build_roi_from_click(const cv::Mat& frame, int x, int y) {
    if (frame.empty()) {
        return {};
    }
    if (x < 0 || y < 0 || x >= frame.cols || y >= frame.rows) {
        return {};
    }

    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame;
    }

    const int search_radius = std::max(1, cfg_.click_search_radius);
    const int x1 = std::max(0, x - search_radius);
    const int y1 = std::max(0, y - search_radius);
    const int x2 = std::min(gray.cols, x + search_radius);
    const int y2 = std::min(gray.rows, y + search_radius);

    cv::Rect search_rect(x1, y1, x2 - x1, y2 - y1);
    if (search_rect.width <= 0 || search_rect.height <= 0) {
        return {};
    }

    cv::Mat search = gray(search_rect);
    cv::Mat search_for_fill = search.clone();
    if (cfg_.click_equalize) {
        cv::equalizeHist(search, search_for_fill);
    }
    cv::Mat mask(search_for_fill.rows + 2, search_for_fill.cols + 2, CV_8UC1, cv::Scalar(0));
    cv::Rect bounding;
    const int flags = 4 | cv::FLOODFILL_MASK_ONLY | (255 << 8);
    const cv::Scalar lo(cfg_.floodfill_lo_diff);
    const cv::Scalar hi(cfg_.floodfill_hi_diff);

    cv::floodFill(search_for_fill, mask, cv::Point(x - search_rect.x, y - search_rect.y),
                  cv::Scalar(255), &bounding, lo, hi, flags);
    if (cfg_.floodfill_fill_overlay) {
        flood_fill_mask_ = cv::Mat::zeros(frame.size(), CV_8UC1);
        flood_fill_overlay_ = cv::Mat::zeros(frame.size(), frame.type());
        cv::Mat mask_roi = mask(cv::Rect(1, 1, search.cols, search.rows));
        mask_roi.copyTo(flood_fill_mask_(search_rect));
        flood_fill_overlay_.setTo(cv::Scalar(0, 0, 255), flood_fill_mask_);
    }

    cv::Rect2f roi(static_cast<float>(bounding.x + search_rect.x),
                   static_cast<float>(bounding.y + search_rect.y),
                   static_cast<float>(bounding.width),
                   static_cast<float>(bounding.height));

    if (roi.area() < static_cast<float>(cfg_.min_area) ||
        roi.width < cfg_.min_width ||
        roi.height < cfg_.min_height) {
        const float half = static_cast<float>(cfg_.fallback_box_size) * 0.5f;
        roi = cv::Rect2f(static_cast<float>(x) - half, static_cast<float>(y) - half,
                         static_cast<float>(cfg_.fallback_box_size),
                         static_cast<float>(cfg_.fallback_box_size));
    }

    roi.x -= cfg_.click_padding;
    roi.y -= cfg_.click_padding;
    roi.width += cfg_.click_padding * 2.0f;
    roi.height += cfg_.click_padding * 2.0f;

    const float max_ratio = std::max(0.0f, std::min(cfg_.max_area_ratio, 1.0f));
    if (max_ratio > 0.0f) {
        const float max_area = static_cast<float>(frame.cols * frame.rows) * max_ratio;
        if (roi.area() > max_area && max_area > 1.0f) {
            const float scale = std::sqrt(max_area / roi.area());
            const cv::Point2f center = rect_center(roi);
            roi.width *= scale;
            roi.height *= scale;
            roi.x = center.x - roi.width * 0.5f;
            roi.y = center.y - roi.height * 0.5f;
        }
    }
    if (!roi.contains(cv::Point2f(static_cast<float>(x), static_cast<float>(y)))) {
        const float half = static_cast<float>(cfg_.fallback_box_size) * 0.5f;
        roi = cv::Rect2f(static_cast<float>(x) - half, static_cast<float>(y) - half,
                         static_cast<float>(cfg_.fallback_box_size),
                         static_cast<float>(cfg_.fallback_box_size));
    }
    return clip_rect(roi, frame.size());
}

cv::Rect2f ManualTrackerManager::find_motion_roi(const cv::Mat& frame, int x, int y) {
    if (prev_gray_.empty() || frame.empty()) {
        return {};
    }

    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame;
    }

    cv::Mat diff;
    cv::absdiff(gray, prev_gray_, diff);
    cv::threshold(diff, diff, cfg_.motion_diff_threshold, 255, cv::THRESH_BINARY);

    const int r = std::max(1, cfg_.motion_search_radius);
    const int x1 = std::max(0, x - r);
    const int y1 = std::max(0, y - r);
    const int x2 = std::min(diff.cols, x + r);
    const int y2 = std::min(diff.rows, y + r);
    if (x2 <= x1 || y2 <= y1) {
        return {};
    }

    cv::Rect search_rect(x1, y1, x2 - x1, y2 - y1);
    cv::Mat search = diff(search_rect);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(search, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return {};
    }

    float best_dist = std::numeric_limits<float>::max();
    cv::Rect best_rect;
    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < cfg_.min_area) {
            continue;
        }
        cv::Rect rect = cv::boundingRect(contour);
        rect.x += search_rect.x;
        rect.y += search_rect.y;
        cv::Point2f center(rect.x + rect.width * 0.5f,
                           rect.y + rect.height * 0.5f);
        const float dx = center.x - static_cast<float>(x);
        const float dy = center.y - static_cast<float>(y);
        const float dist = std::sqrt(dx * dx + dy * dy);
        if (dist < best_dist) {
            best_dist = dist;
            best_rect = rect;
        }
    }

    if (best_rect.area() <= 0) {
        return {};
    }
    return clip_rect(cv::Rect2f(best_rect), frame.size());
}

void ManualTrackerManager::init_kalman(ManualTrack& track, const cv::Point2f& center) {
    track.kf = cv::KalmanFilter(4, 2, 0, CV_32F);
    track.kf.transitionMatrix = (cv::Mat_<float>(4, 4) <<
                                                       1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1);
    track.kf.measurementMatrix = (cv::Mat_<float>(2, 4) <<
                                                        1, 0, 0, 0,
            0, 1, 0, 0);
    track.kf.processNoiseCov = cv::Mat::eye(4, 4, CV_32F) * cfg_.kalman_process_noise;
    track.kf.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * cfg_.kalman_measurement_noise;
    track.kf.errorCovPost = cv::Mat::eye(4, 4, CV_32F);
    track.kf.statePost = (cv::Mat_<float>(4, 1) << center.x, center.y, 0.0f, 0.0f);
    track.kf_ready = true;
}

void ManualTrackerManager::predict_kalman(ManualTrack& track) {
    if (!track.kf_ready) {
        return;
    }
    cv::Mat prediction = track.kf.predict();
    float px = prediction.at<float>(0);
    float py = prediction.at<float>(1);
    track.bbox.x = px - track.bbox.width * 0.5f;
    track.bbox.y = py - track.bbox.height * 0.5f;
}

void ManualTrackerManager::correct_kalman(ManualTrack& track, const cv::Point2f& center) {
    if (!track.kf_ready) {
        return;
    }
    cv::Mat measurement = (cv::Mat_<float>(2, 1) << center.x, center.y);
    track.kf.correct(measurement);
}

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

bool ManualTrackerManager::try_reacquire_with_template(ManualTrack& track, const cv::Mat& frame) {
    if (track.template_gray.empty()) {
        return false;
    }

    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame;
    }

    cv::Point2f center = rect_center(track.bbox);
    int x1 = std::max(0, static_cast<int>(center.x) - cfg_.search_radius);
    int y1 = std::max(0, static_cast<int>(center.y) - cfg_.search_radius);
    int x2 = std::min(gray.cols, static_cast<int>(center.x) + cfg_.search_radius);
    int y2 = std::min(gray.rows, static_cast<int>(center.y) + cfg_.search_radius);
    if (x2 <= x1 || y2 <= y1) {
        return false;
    }

    cv::Rect search_rect(x1, y1, x2 - x1, y2 - y1);
    cv::Mat search = gray(search_rect);

    if (search.cols < track.template_gray.cols || search.rows < track.template_gray.rows) {
        return false;
    }

    cv::Mat result;
    cv::matchTemplate(search, track.template_gray, result, cv::TM_CCOEFF_NORMED);
    double min_val = 0.0;
    double max_val = 0.0;
    cv::Point min_loc;
    cv::Point max_loc;
    cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);

    if (max_val < cfg_.match_threshold) {
        return false;
    }

    cv::Rect2f new_box(
            static_cast<float>(search_rect.x + max_loc.x),
            static_cast<float>(search_rect.y + max_loc.y),
            static_cast<float>(track.template_gray.cols),
            static_cast<float>(track.template_gray.rows)
    );
    track.bbox = clip_rect(new_box, frame.size());
    if (track.bbox.area() <= 1.0f) {
        return false;
    }
    track.tracker = create_tracker();
    track.tracker->init(frame, track.bbox);
    track.predicted = false;
    track.missed_frames = 0;
    correct_kalman(track, rect_center(track.bbox));
    if (cfg_.update_template) {
        track.template_gray = gray(to_int_rect(track.bbox)).clone();
    }
    track.logged_reacquire_ready = false;
    if (log_cfg_.tracker_level_logger) {
        std::cout << "[TRK] reacquire by template id=" << track.id << std::endl;
    }
    return true;
}

float ManualTrackerManager::compute_contrast(const cv::Mat& frame, const cv::Rect2f& roi) const {
    if (frame.empty() || roi.area() <= 1.0f) {
        return 0.0f;
    }
    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame;
    }
    cv::Rect roi_int = to_int_rect(clip_rect(roi, frame.size()));
    if (roi_int.width <= 0 || roi_int.height <= 0) {
        return 0.0f;
    }
    cv::Mat roi_mat = gray(roi_int);
    cv::Scalar mean, stddev;
    cv::meanStdDev(roi_mat, mean, stddev);
    return static_cast<float>(stddev[0]);
}

bool ManualTrackerManager::handle_click(int x, int y, const cv::Mat& frame, long long now_ms) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto it = tracks_.begin(); it != tracks_.end(); ++it) {
        if (point_in_rect_with_padding(it->bbox, x, y, cfg_.remove_padding)) {
            tracks_.erase(it);
            refresh_targets();
            return true;
        }
    }

    if (static_cast<int>(tracks_.size()) >= cfg_.max_targets) {
        return false;
    }

    cv::Rect2f roi = find_motion_roi(frame, x, y);
    bool is_dynamic = roi.area() > 1.0f;
    if (roi.area() <= 1.0f) {
        roi = build_roi_from_click(frame, x, y);
    }
    if (roi.area() <= 1.0f) {
        return false;
    }
    const float contrast = compute_contrast(frame, roi);
    if (log_cfg_.manual_detector_level_logger) {
        if (is_dynamic) {
            std::cout << "[MANUAL] capture dynamic click id=" << next_id_
                      << " contrast=" << contrast << std::endl;
        } else {
            std::cout << "[MANUAL] capture static click id=" << next_id_
                      << " contrast=" << contrast << std::endl;
        }
    }

    ManualTrack track;
    track.id = next_id_++;
    track.bbox = roi;
    track.is_dynamic = is_dynamic;
    track.tracker = create_tracker();
    track.tracker->init(frame, roi);

    cv::Mat gray;
    if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame;
    }
    track.template_gray = gray(to_int_rect(roi)).clone();

    track.last_seen_ms = now_ms;
    track.missed_frames = 0;
    track.predicted = false;
    init_kalman(track, rect_center(track.bbox));

    tracks_.push_back(std::move(track));
    refresh_targets();
    if (log_cfg_.tracker_level_logger) {
        std::cout << "[TRK] start "
                  << (is_dynamic ? "dynamic" : "static")
                  << " id=" << (next_id_ - 1) << std::endl;
    }
    if (log_cfg_.target_object_created_logger) {
        std::cout << "[MANUAL] target object created id=" << (next_id_ - 1)
                  << " contrast=" << contrast << std::endl;
    }
    return true;
}

void ManualTrackerManager::update(cv::Mat& frame, long long now_ms) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (cfg_.auto_reacquire_nearest) {
        for (auto& track : tracks_) {
            if (track.missed_frames == 0) {
                continue;
            }
            const long long lost_ms = now_ms - track.last_seen_ms;
            if (lost_ms < cfg_.reacquire_delay_ms) {
                continue;
            }
            if (log_cfg_.tracker_level_logger && !track.logged_reacquire_ready) {
                std::cout << "[TRK] reacquire ready id=" << track.id << std::endl;
                track.logged_reacquire_ready = true;
            }

            float best_dist = std::numeric_limits<float>::max();
            ManualTrack* best = nullptr;
            const cv::Point2f lost_center = rect_center(track.bbox);
            for (auto& candidate : tracks_) {
                if (candidate.id == track.id || candidate.missed_frames > 0) {
                    continue;
                }
                const cv::Point2f cand_center = rect_center(candidate.bbox);
                const float dx = lost_center.x - cand_center.x;
                const float dy = lost_center.y - cand_center.y;
                const float dist = std::sqrt(dx * dx + dy * dy);
                if (dist < best_dist) {
                    best_dist = dist;
                    best = &candidate;
                }
            }

            if (best && best_dist <= static_cast<float>(cfg_.reacquire_max_distance_px)) {
                track.bbox = best->bbox;
                track.tracker = create_tracker();
                track.tracker->init(frame, track.bbox);
                track.template_gray = best->template_gray.clone();
                track.missed_frames = 0;
                track.predicted = false;
                track.last_seen_ms = now_ms;
                correct_kalman(track, rect_center(track.bbox));
                track.logged_reacquire_ready = false;
                if (log_cfg_.tracker_level_logger) {
                    std::cout << "[TRK] reacquire nearest id=" << track.id << std::endl;
                }
            }
        }
    }

    for (auto it = tracks_.begin(); it != tracks_.end(); ) {
        it->age_frames += 1;
        bool updated = false;
        if (it->tracker) {
            cv::Rect new_box;
            if (it->tracker->update(frame, new_box)) {
                it->bbox = clip_rect(cv::Rect2f(new_box), frame.size());
                it->missed_frames = 0;
                it->predicted = false;
                it->last_seen_ms = now_ms;
                it->logged_reacquire_ready = false;
                correct_kalman(*it, rect_center(it->bbox));
                if (cfg_.update_template) {
                    cv::Mat gray;
                    if (frame.channels() == 3) {
                        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                    } else {
                        gray = frame;
                    }
                    it->template_gray = gray(to_int_rect(it->bbox)).clone();
                }
                updated = true;
            }
        }

        if (!updated) {
            it->missed_frames += 1;
            it->predicted = true;
            predict_kalman(*it);
            try_reacquire_with_template(*it, frame);
        }

        if (now_ms - it->last_seen_ms > cfg_.max_lost_ms) {
            if (log_cfg_.tracker_level_logger) {
                std::cout << "[TRK] lost "
                          << (it->is_dynamic ? "dynamic" : "static")
                          << " id=" << it->id << std::endl;
            }
            it = tracks_.erase(it);
            continue;
        }
        ++it;
    }

    refresh_targets();

    if (frame.empty()) {
        return;
    }
    if (cfg_.floodfill_fill_overlay && !flood_fill_overlay_.empty() && !flood_fill_mask_.empty()) {
        cv::Mat blended;
        constexpr double kOverlayAlpha = 0.7;
        cv::addWeighted(frame, 1.0 - kOverlayAlpha, flood_fill_overlay_, kOverlayAlpha, 0.0, blended);
        blended.copyTo(frame, flood_fill_mask_);
    }
    if (frame.channels() == 3) {
        cv::cvtColor(frame, prev_gray_, cv::COLOR_BGR2GRAY);
    } else {
        prev_gray_ = frame.clone();
    }
}

void ManualTrackerManager::refresh_targets() {
    targets_.clear();
    targets_.reserve(tracks_.size());
    for (const auto& tr : tracks_) {
        Target tg;
        tg.id = tr.id;
        tg.target_name = "T" + std::to_string(tr.id);
        tg.bbox = tr.bbox;
        tg.age_frames = tr.age_frames;
        tg.missed_frames = tr.missed_frames;
        targets_.push_back(std::move(tg));
    }
}
