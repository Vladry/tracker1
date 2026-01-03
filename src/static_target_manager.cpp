#include "static_target_manager.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace {
    // Округляет прямоугольник с float-координатами до int.
    static inline cv::Rect to_int_rect(const cv::Rect2f& rect) {
        return cv::Rect(
                static_cast<int>(std::round(rect.x)),
                static_cast<int>(std::round(rect.y)),
                static_cast<int>(std::round(rect.width)),
                static_cast<int>(std::round(rect.height))
        );
    }
}

StaticTargetManager::StaticTargetManager(const toml::table& tbl) {
    load_logging_config(tbl, log_cfg_);
    load_config(tbl);
}

bool StaticTargetManager::load_config(const toml::table& tbl) {
    try {
        const auto *cfg = tbl["static_detector"].as_table();
        if (!cfg) {
            throw std::runtime_error("missing [static_detector] table");
        }
        cfg_.max_targets = read_required<int>(*cfg, "max_targets");
        cfg_.click_padding = read_required<int>(*cfg, "click_padding");
        cfg_.fallback_box_size = read_required<int>(*cfg, "fallback_box_size");
        cfg_.max_area_ratio = read_required<float>(*cfg, "max_area_ratio");
        cfg_.click_equalize = read_required<bool>(*cfg, "click_equalize");
        cfg_.floodfill_lo_diff = read_required<int>(*cfg, "floodfill_lo_diff");
        cfg_.floodfill_hi_diff = read_required<int>(*cfg, "floodfill_hi_diff");
        cfg_.overlay_ttl_seconds = read_required<int>(*cfg, "overlay_ttl_seconds");
        cfg_.min_area = read_required<int>(*cfg, "min_area");
        cfg_.min_width = read_required<int>(*cfg, "min_width");
        cfg_.min_height = read_required<int>(*cfg, "min_height");
        cfg_.min_contrast = read_required<float>(*cfg, "min_contrast");
        std::cout << "[STATIC] config: max_targets=" << cfg_.max_targets
                  << " click_padding=" << cfg_.click_padding
                  << " remove_padding=" << cfg_.remove_padding
                  << " fallback_box_size=" << cfg_.fallback_box_size
                  << " max_area_ratio=" << cfg_.max_area_ratio
                  << " click_equalize=" << (cfg_.click_equalize ? "true" : "false")
                  << " floodfill_lo_diff=" << cfg_.floodfill_lo_diff
                  << " floodfill_hi_diff=" << cfg_.floodfill_hi_diff
                  << " overlay_ttl_seconds=" << cfg_.overlay_ttl_seconds
                  << " min_area=" << cfg_.min_area
                  << " min_width=" << cfg_.min_width
                  << " min_height=" << cfg_.min_height
                  << " min_contrast=" << cfg_.min_contrast
                  << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[STATIC] config load failed: " << e.what() << std::endl;
        return false;
    }
}

cv::Rect2f StaticTargetManager::clip_rect(const cv::Rect2f& rect, const cv::Size& size) const {
    float x1 = std::max(0.0f, rect.x);
    float y1 = std::max(0.0f, rect.y);
    float x2 = std::min(rect.x + rect.width, static_cast<float>(size.width));
    float y2 = std::min(rect.y + rect.height, static_cast<float>(size.height));
    if (x2 <= x1 || y2 <= y1) {
        return {};
    }
    return {x1, y1, x2 - x1, y2 - y1};
}

bool StaticTargetManager::point_in_rect_with_padding(const cv::Rect2f& rect, int x, int y, int pad) const {
    cv::Rect2f padded(rect.x - pad, rect.y - pad, rect.width + pad * 2.0f, rect.height + pad * 2.0f);
    return padded.contains(cv::Point2f(static_cast<float>(x), static_cast<float>(y)));
}

cv::Rect2f StaticTargetManager::build_roi_from_click(const cv::Mat& frame, int x, int y, cv::Mat* mask_out) const {
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

    cv::Rect search_rect(0, 0, gray.cols, gray.rows);
    if (search_rect.width <= 0 || search_rect.height <= 0) {
        return {};
    }

    cv::Mat search = gray(search_rect);
    cv::Mat search_for_fill = search.clone();
    if (cfg_.click_equalize) {
        cv::equalizeHist(search, search_for_fill);
    }

    cv::Mat flood_mask(search_for_fill.rows + 2, search_for_fill.cols + 2, CV_8UC1, cv::Scalar(0));
    cv::Rect bounding;
    const int flags = 4 | cv::FLOODFILL_MASK_ONLY | (255 << 8);
    const cv::Scalar lo(cfg_.floodfill_lo_diff);
    const cv::Scalar hi(cfg_.floodfill_hi_diff);

    cv::floodFill(search_for_fill, flood_mask, cv::Point(x - search_rect.x, y - search_rect.y),
                  cv::Scalar(255), &bounding, lo, hi, flags);

    if (mask_out) {
        cv::Mat full_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
        cv::Mat mask_roi = flood_mask(cv::Rect(1, 1, search.cols, search.rows));
        mask_roi.copyTo(full_mask(search_rect));
        *mask_out = full_mask;
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
            const cv::Point2f center(roi.x + roi.width * 0.5f, roi.y + roi.height * 0.5f);
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

float StaticTargetManager::compute_contrast(const cv::Mat& frame, const cv::Rect2f& roi) const {
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

bool StaticTargetManager::handle_right_click(int x, int y, const cv::Mat& frame, long long now_ms) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto it = targets_.begin(); it != targets_.end(); ++it) {
        if (point_in_rect_with_padding(it->bbox, x, y, cfg_.remove_padding)) {
            const int removed_id = it->id;
            targets_.erase(it);
            if (log_cfg_.manual_detector_level_logger) {
                std::cout << "[STATIC] remove id=" << removed_id << std::endl;
            }
            return true;
        }
    }

    if (static_cast<int>(targets_.size()) >= cfg_.max_targets) {
        return false;
    }

    if (frame.empty()) {
        return false;
    }

    cv::Mat flood_mask;
    cv::Rect2f roi = build_roi_from_click(frame, x, y, &flood_mask);
    if (roi.area() <= 1.0f) {
        return false;
    }

    const float contrast = compute_contrast(frame, roi);
    if (contrast < cfg_.min_contrast) {
        if (log_cfg_.manual_detector_level_logger) {
            std::cout << "[STATIC] click ignored: low contrast=" << contrast << std::endl;
        }
        return false;
    }

    StaticTarget target;
    target.id = next_id_++;
    target.bbox = roi;
    targets_.push_back(std::move(target));

    if (!flood_mask.empty()) {
        flood_fill_mask_ = flood_mask;
        flood_fill_overlay_ = cv::Mat::zeros(frame.size(), frame.type());
        flood_fill_overlay_.setTo(cv::Scalar(0, 255, 255), flood_fill_mask_);
        overlay_expire_ms_ = now_ms + static_cast<long long>(cfg_.overlay_ttl_seconds) * 1000;
    }

    if (log_cfg_.manual_detector_level_logger) {
        std::cout << "[STATIC] capture id=" << (next_id_ - 1)
                  << " contrast=" << contrast << std::endl;
    }
    return true;
}

void StaticTargetManager::update(cv::Mat& frame, long long now_ms) {
    if (overlay_expire_ms_ > 0 && now_ms >= overlay_expire_ms_) {
        flood_fill_overlay_.release();
        flood_fill_mask_.release();
        overlay_expire_ms_ = 0;
    }
    if (!flood_fill_overlay_.empty() && !flood_fill_mask_.empty()) {
        cv::Mat blended;
        constexpr double kOverlayAlpha = 0.7;
        cv::addWeighted(frame, 1.0 - kOverlayAlpha, flood_fill_overlay_, kOverlayAlpha, 0.0, blended);
        blended.copyTo(frame, flood_fill_mask_);
    }
}
