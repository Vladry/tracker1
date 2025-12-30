#include "manual_tracker_manager.h"
#include <algorithm>
#include <cmath>
#include <iostream>
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
        cfg_.floodfill_lo_diff = read_required<int>(*cfg, "floodfill_lo_diff");
        cfg_.floodfill_hi_diff = read_required<int>(*cfg, "floodfill_hi_diff");
        cfg_.min_area = read_required<int>(*cfg, "min_area");
        cfg_.min_width = read_required<int>(*cfg, "min_width");
        cfg_.min_height = read_required<int>(*cfg, "min_height");
        cfg_.search_radius = read_required<int>(*cfg, "search_radius");
        cfg_.match_threshold = read_required<float>(*cfg, "match_threshold");
        cfg_.update_template = read_required<bool>(*cfg, "update_template");
        cfg_.max_lost_ms = read_required<int>(*cfg, "max_lost_ms");
        cfg_.kalman_process_noise = read_required<float>(*cfg, "kalman_process_noise");
        cfg_.kalman_measurement_noise = read_required<float>(*cfg, "kalman_measurement_noise");
        std::cout << "[MANUAL] config: max_targets=" << cfg_.max_targets
                  << " click_padding=" << cfg_.click_padding
                  << " remove_padding=" << cfg_.remove_padding
                  << " fallback_box_size=" << cfg_.fallback_box_size
                  << " floodfill_lo_diff=" << cfg_.floodfill_lo_diff
                  << " floodfill_hi_diff=" << cfg_.floodfill_hi_diff
                  << " min_area=" << cfg_.min_area
                  << " min_width=" << cfg_.min_width
                  << " min_height=" << cfg_.min_height
                  << " search_radius=" << cfg_.search_radius
                  << " match_threshold=" << cfg_.match_threshold
                  << " update_template=" << (cfg_.update_template ? "true" : "false")
                  << " max_lost_ms=" << cfg_.max_lost_ms
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

cv::Rect2f ManualTrackerManager::build_roi_from_click(const cv::Mat& frame, int x, int y) const {
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

    cv::Mat mask(gray.rows + 2, gray.cols + 2, CV_8UC1, cv::Scalar(0));
    cv::Rect bounding;
    const int flags = 4 | cv::FLOODFILL_MASK_ONLY | (255 << 8);
    const cv::Scalar lo(cfg_.floodfill_lo_diff);
    const cv::Scalar hi(cfg_.floodfill_hi_diff);

    cv::floodFill(gray, mask, cv::Point(x, y), cv::Scalar(255), &bounding, lo, hi, flags);

    cv::Rect2f roi(static_cast<float>(bounding.x),
                   static_cast<float>(bounding.y),
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
    return clip_rect(roi, frame.size());
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
    track.tracker = cv::TrackerCSRT::create();
    track.tracker->init(frame, track.bbox);
    track.predicted = false;
    track.missed_frames = 0;
    correct_kalman(track, rect_center(track.bbox));
    if (cfg_.update_template) {
        track.template_gray = gray(to_int_rect(track.bbox)).clone();
    }
    return true;
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

    cv::Rect2f roi = build_roi_from_click(frame, x, y);
    if (roi.area() <= 1.0f) {
        return false;
    }

    ManualTrack track;
    track.id = next_id_++;
    track.bbox = roi;
    track.tracker = cv::TrackerCSRT::create();
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
    return true;
}

void ManualTrackerManager::update(const cv::Mat& frame, long long now_ms) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto it = tracks_.begin(); it != tracks_.end(); ) {
        it->age_frames += 1;
        bool updated = false;
        if (it->tracker) {
            cv::Rect2d new_box;
            if (it->tracker->update(frame, new_box)) {
                it->bbox = clip_rect(cv::Rect2f(new_box), frame.size());
                it->missed_frames = 0;
                it->predicted = false;
                it->last_seen_ms = now_ms;
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
            it = tracks_.erase(it);
            continue;
        }
        ++it;
    }

    refresh_targets();
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
