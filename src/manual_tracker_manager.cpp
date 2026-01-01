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

    constexpr int kReacquireNone = 0;
    constexpr int kReacquirePriority1 = 1;
    constexpr int kReacquirePriority2 = 2;
    constexpr int kReacquirePriority3 = 3;
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
        cfg_.tracker_motion_min_ratio = read_required<float>(*cfg, "tracker_motion_min_ratio");
        cfg_.tracker_motion_grace_frames = read_required<int>(*cfg, "tracker_motion_grace_frames");
        cfg_.lost_bbox_ttl_ms = read_required<int>(*cfg, "lost_bbox_ttl_ms");
        cfg_.reacquire_fallback_max_distance_px = read_required<int>(*cfg, "reacquire_fallback_max_distance_px");
        cfg_.reacquire_kalman_radius_px = read_required<int>(*cfg, "reacquire_kalman_radius_px");
        cfg_.reacquire_near_radius_px = read_required<int>(*cfg, "reacquire_near_radius_px");
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
                  << " tracker_motion_min_ratio=" << cfg_.tracker_motion_min_ratio
                  << " tracker_motion_grace_frames=" << cfg_.tracker_motion_grace_frames
                  << " lost_bbox_ttl_ms=" << cfg_.lost_bbox_ttl_ms
                  << " reacquire_fallback_max_distance_px=" << cfg_.reacquire_fallback_max_distance_px
                  << " reacquire_kalman_radius_px=" << cfg_.reacquire_kalman_radius_px
                  << " reacquire_near_radius_px=" << cfg_.reacquire_near_radius_px
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

cv::Rect ManualTrackerManager::make_centered_roi(const cv::Point2f& center, int size, const cv::Size& frame_size) const {
    const int half = size / 2;
    int x1 = static_cast<int>(std::round(center.x)) - half;
    int y1 = static_cast<int>(std::round(center.y)) - half;
    x1 = std::max(0, x1);
    y1 = std::max(0, y1);
    int x2 = std::min(frame_size.width, x1 + size);
    int y2 = std::min(frame_size.height, y1 + size);
    if (x2 <= x1 || y2 <= y1) {
        return {};
    }
    return {x1, y1, x2 - x1, y2 - y1};
}

std::vector<cv::Rect> ManualTrackerManager::find_motion_clusters(
        const cv::Mat& current_gray,
        const cv::Mat& prev_gray,
        const cv::Rect& search_rect) const {
    std::vector<cv::Rect> clusters;
    if (current_gray.empty() || prev_gray.empty() || search_rect.area() <= 0) {
        return clusters;
    }
    cv::Rect clipped = search_rect & cv::Rect(0, 0, current_gray.cols, current_gray.rows);
    if (clipped.area() <= 0) {
        return clusters;
    }
    cv::Mat diff;
    cv::absdiff(current_gray(clipped), prev_gray(clipped), diff);
    cv::threshold(diff, diff, cfg_.motion_diff_threshold, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(diff, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < cfg_.min_area) {
            continue;
        }
        cv::Rect rect = cv::boundingRect(contour);
        rect.x += clipped.x;
        rect.y += clipped.y;
        clusters.push_back(rect);
    }
    return clusters;
}

bool ManualTrackerManager::try_reacquire_with_motion(
        ManualTrack& track,
        const cv::Mat& gray,
        const cv::Mat& frame,
        long long now_ms,
        int stage,
        const cv::Rect& roi,
        const char* log_label) {
    if (frame.empty() || gray.empty() || roi.area() <= 0) {
        return false;
    }

    if (track.reacquire_stage != stage ||
        track.reacquire_roi.x != roi.x ||
        track.reacquire_roi.y != roi.y ||
        track.reacquire_roi.width != roi.width ||
        track.reacquire_roi.height != roi.height) {
        track.reacquire_stage = stage;
        track.reacquire_roi = roi;
        track.reacquire_gray_frames.clear();
    }

    track.reacquire_gray_frames.push_back(gray);
    const int required = std::max(1, cfg_.motion_frames) + 1;
    if (static_cast<int>(track.reacquire_gray_frames.size()) < required) {
        return false;
    }
    if (static_cast<int>(track.reacquire_gray_frames.size()) > required) {
        track.reacquire_gray_frames.erase(track.reacquire_gray_frames.begin());
    }

    std::vector<cv::Point2f> motion_points;
    cv::Rect2f motion_roi = build_motion_roi_from_sequence(track.reacquire_gray_frames, roi, motion_points);
    if (motion_roi.area() <= 1.0f) {
        motion_roi = build_motion_roi_from_diff(track.reacquire_gray_frames, roi);
    }

    if (motion_roi.area() <= 1.0f ||
        motion_roi.area() < static_cast<float>(cfg_.min_area) ||
        motion_roi.width < cfg_.min_width ||
        motion_roi.height < cfg_.min_height) {
        return false;
    }

    motion_roi.x -= cfg_.click_padding;
    motion_roi.y -= cfg_.click_padding;
    motion_roi.width += cfg_.click_padding * 2.0f;
    motion_roi.height += cfg_.click_padding * 2.0f;
    motion_roi = clip_rect(motion_roi, frame.size());
    if (motion_roi.area() <= 1.0f) {
        return false;
    }

    cv::Rect2f tracker_roi = expand_rect(motion_roi, static_cast<float>(cfg_.tracker_init_padding));
    tracker_roi = ensure_min_size(tracker_roi, static_cast<float>(cfg_.tracker_min_size));
    tracker_roi = clip_rect(tracker_roi, frame.size());
    if (tracker_roi.area() <= 1.0f) {
        return false;
    }

    track.bbox = tracker_roi;
    track.tracker = create_tracker();
    track.tracker->init(frame, track.bbox);
    track.template_gray = gray(to_int_rect(track.bbox)).clone();
    track.missed_frames = 0;
    track.predicted = false;
    track.predicted_center_ready = false;
    track.last_seen_ms = now_ms;
    track.logged_reacquire_ready = false;
    track.stale_frames = 0;
    track.lost_since_ms = 0;
    track.reacquire_gray_frames.clear();
    track.reacquire_stage = 0;
    correct_kalman(track, rect_center(track.bbox));

    if (log_cfg_.reacquire_level_logger) {
        std::cout << log_label << " id=" << track.id << std::endl;
    }
    return true;
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

// ... (оставшаяся часть файла без изменений)
