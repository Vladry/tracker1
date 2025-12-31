#include "manual_tracker_manager.h"
#include <algorithm>
#include <cmath>
#include <cctype>
#include <iostream>
#include <limits>
#include <map>
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

    static inline cv::Mat to_gray(const cv::Mat& frame) {
        if (frame.channels() == 3) {
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            return gray;
        }
        return frame.clone();
    }

    static inline cv::Rect2f expand_rect(const cv::Rect2f& rect, float pad) {
        return cv::Rect2f(rect.x - pad, rect.y - pad, rect.width + pad * 2.0f, rect.height + pad * 2.0f);
    }

    static inline cv::Rect2f ensure_min_size(const cv::Rect2f& rect, float min_size) {
        if (rect.width >= min_size && rect.height >= min_size) {
            return rect;
        }
        const cv::Point2f center(rect.x + rect.width * 0.5f, rect.y + rect.height * 0.5f);
        const float size = std::max(min_size, std::max(rect.width, rect.height));
        return cv::Rect2f(center.x - size * 0.5f, center.y - size * 0.5f, size, size);
    }

    static inline float angle_between(float a, float b) {
        constexpr float kPi = 3.14159265f;
        float diff = std::fabs(a - b);
        if (diff > kPi) {
            diff = 2.0f * kPi - diff;
        }
        return diff;
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
        cfg_.click_capture_size = read_required<int>(*cfg, "click_capture_size");
        cfg_.motion_frames = read_required<int>(*cfg, "motion_frames");
        cfg_.overlay_ttl_seconds = read_required<int>(*cfg, "overlay_ttl_seconds");
        cfg_.tracker_init_padding = read_required<int>(*cfg, "tracker_init_padding");
        cfg_.tracker_min_size = read_required<int>(*cfg, "tracker_min_size");
        cfg_.motion_min_magnitude = read_required<float>(*cfg, "motion_min_magnitude");
        cfg_.motion_angle_tolerance_deg = read_required<float>(*cfg, "motion_angle_tolerance_deg");
        cfg_.motion_mag_tolerance_px = read_required<float>(*cfg, "motion_mag_tolerance_px");
        cfg_.lost_bbox_ttl_ms = read_required<int>(*cfg, "lost_bbox_ttl_ms");
        cfg_.reacquire_fallback_max_distance_px = read_required<int>(*cfg, "reacquire_fallback_max_distance_px");
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
                  << " click_capture_size=" << cfg_.click_capture_size
                  << " motion_frames=" << cfg_.motion_frames
                  << " overlay_ttl_seconds=" << cfg_.overlay_ttl_seconds
                  << " tracker_init_padding=" << cfg_.tracker_init_padding
                  << " tracker_min_size=" << cfg_.tracker_min_size
                  << " motion_min_magnitude=" << cfg_.motion_min_magnitude
                  << " motion_angle_tolerance_deg=" << cfg_.motion_angle_tolerance_deg
                  << " motion_mag_tolerance_px=" << cfg_.motion_mag_tolerance_px
                  << " lost_bbox_ttl_ms=" << cfg_.lost_bbox_ttl_ms
                  << " reacquire_fallback_max_distance_px=" << cfg_.reacquire_fallback_max_distance_px
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

cv::Rect ManualTrackerManager::make_click_roi(const cv::Mat& frame, int x, int y) const {
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

cv::Rect2f ManualTrackerManager::build_motion_roi_from_sequence(
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
    track.predicted_center = cv::Point2f(px, py);
    track.predicted_center_ready = true;
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
    if (track.predicted_center_ready) {
        center = track.predicted_center;
    }
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
        if (log_cfg_.reacquire_level_logger) {
            std::cout << "[TRK] priority1 (kalman) template miss id=" << track.id
                      << " score=" << max_val << std::endl;
        }
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
    track.predicted_center_ready = false;
    correct_kalman(track, rect_center(track.bbox));
    if (cfg_.update_template) {
        track.template_gray = gray(to_int_rect(track.bbox)).clone();
    }
    track.logged_reacquire_ready = false;
    if (log_cfg_.reacquire_level_logger) {
        std::cout << "[TRK] priority1 (kalman) template reacquire id=" << track.id
                  << " score=" << max_val << std::endl;
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

    if (frame.empty()) {
        return false;
    }

    cv::Rect roi = make_click_roi(frame, x, y);
    if (roi.area() <= 0) {
        return false;
    }

    PendingClick pending;
    pending.click = {x, y};
    pending.roi = roi;
    pending.start_ms = now_ms;
    pending.gray_frames.push_back(to_gray(frame));
    pending_clicks_.push_back(std::move(pending));
    if (log_cfg_.manual_detector_level_logger) {
        std::cout << "[MANUAL] pending motion click id=" << next_id_
                  << " roi=" << roi.width << "x" << roi.height << std::endl;
    }
    return true;
}

void ManualTrackerManager::update(cv::Mat& frame, long long now_ms) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!pending_clicks_.empty() && !frame.empty()) {
        cv::Mat gray = to_gray(frame);
        for (auto it = pending_clicks_.begin(); it != pending_clicks_.end(); ) {
            it->gray_frames.push_back(gray);
            const int required = std::max(1, cfg_.motion_frames) + 1;
            if (static_cast<int>(it->gray_frames.size()) < required) {
                ++it;
                continue;
            }

            if (static_cast<int>(tracks_.size()) >= cfg_.max_targets) {
                it = pending_clicks_.erase(it);
                continue;
            }

            std::vector<cv::Point2f> motion_points;
            cv::Rect2f roi = build_motion_roi_from_sequence(it->gray_frames, it->roi, motion_points);
            if (roi.area() > 1.0f) {
                roi.x -= cfg_.click_padding;
                roi.y -= cfg_.click_padding;
                roi.width += cfg_.click_padding * 2.0f;
                roi.height += cfg_.click_padding * 2.0f;
                roi = clip_rect(roi, frame.size());
            }

            if (roi.area() > 1.0f &&
                roi.area() >= static_cast<float>(cfg_.min_area) &&
                roi.width >= cfg_.min_width &&
                roi.height >= cfg_.min_height) {
                cv::Rect2f tracker_roi = roi;
                tracker_roi = expand_rect(tracker_roi, static_cast<float>(cfg_.tracker_init_padding));
                tracker_roi = ensure_min_size(tracker_roi, static_cast<float>(cfg_.tracker_min_size));
                tracker_roi = clip_rect(tracker_roi, frame.size());
                if (tracker_roi.area() <= 1.0f) {
                    it = pending_clicks_.erase(it);
                    continue;
                }

                const float contrast = compute_contrast(frame, tracker_roi);
                if (log_cfg_.manual_detector_level_logger) {
                    std::cout << "[MANUAL] capture dynamic click id=" << next_id_
                              << " contrast=" << contrast << std::endl;
                }

                ManualTrack track;
                track.id = next_id_++;
                track.bbox = tracker_roi;
                track.is_dynamic = true;
                track.tracker = create_tracker();
                track.tracker->init(frame, track.bbox);
                track.template_gray = gray(to_int_rect(track.bbox)).clone();
                track.last_seen_ms = now_ms;
                track.missed_frames = 0;
                track.predicted = false;
                track.predicted_center_ready = false;
                init_kalman(track, rect_center(track.bbox));

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
                        cv::Rect overlay_rect = to_int_rect(roi);
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
                    std::cout << "[MANUAL] target object created id=" << (next_id_ - 1)
                              << " contrast=" << contrast << std::endl;
                }
            } else if (log_cfg_.manual_detector_level_logger) {
                std::cout << "[MANUAL] motion click ignored (static) id=" << next_id_ << std::endl;
            }
            it = pending_clicks_.erase(it);
        }
    }

    if (cfg_.auto_reacquire_nearest) {
        for (auto& track : tracks_) {
            if (track.missed_frames == 0) {
                continue;
            }
            const long long lost_ms = now_ms - track.last_seen_ms;
            if (lost_ms < cfg_.reacquire_delay_ms) {
                continue;
            }
            if (log_cfg_.reacquire_level_logger && !track.logged_reacquire_ready) {
                std::cout << "[TRK] priority1 (kalman wait) ready id=" << track.id << std::endl;
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
                if (log_cfg_.reacquire_level_logger) {
                    std::cout << "[TRK] priority2 (nearest) reacquired id=" << track.id
                              << " dist=" << best_dist << std::endl;
                }
                continue;
            }

            if (best && best_dist <= static_cast<float>(cfg_.reacquire_fallback_max_distance_px)) {
                track.bbox = best->bbox;
                track.tracker = create_tracker();
                track.tracker->init(frame, track.bbox);
                track.template_gray = best->template_gray.clone();
                track.missed_frames = 0;
                track.predicted = false;
                track.last_seen_ms = now_ms;
                correct_kalman(track, rect_center(track.bbox));
                track.logged_reacquire_ready = false;
                if (log_cfg_.reacquire_level_logger) {
                    std::cout << "[TRK] priority3 (fallback) reacquired id=" << track.id
                              << " dist=" << best_dist << std::endl;
                }
            } else if (log_cfg_.reacquire_level_logger) {
                std::cout << "[TRK] priority3 (fallback) no candidates id=" << track.id << std::endl;
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
                it->predicted_center_ready = false;
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

        const long long lost_ttl_ms = cfg_.lost_bbox_ttl_ms > 0 ? cfg_.lost_bbox_ttl_ms : cfg_.max_lost_ms;
        if (now_ms - it->last_seen_ms > lost_ttl_ms) {
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
