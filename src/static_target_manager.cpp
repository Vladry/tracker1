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
        cfg_.MAX_TARGETS = read_required<int>(*cfg, "MAX_TARGETS");
        cfg_.CLICK_PADDING = read_required<int>(*cfg, "CLICK_PADDING");
        cfg_.REMOVE_PADDING = read_required<int>(*cfg, "REMOVE_PADDING");
        cfg_.FALLBACK_BOX_SIZE = read_required<int>(*cfg, "FALLBACK_BOX_SIZE");
        cfg_.MAX_AREA_RATIO = read_required<float>(*cfg, "MAX_AREA_RATIO");
        cfg_.CLICK_EQUALIZE = read_required<bool>(*cfg, "CLICK_EQUALIZE");
        cfg_.FLOODFILL_LO_DIFF = read_required<int>(*cfg, "FLOODFILL_LO_DIFF");
        cfg_.FLOODFILL_HI_DIFF = read_required<int>(*cfg, "FLOODFILL_HI_DIFF");
        cfg_.OVERLAY_TTL_SECONDS = read_required<int>(*cfg, "OVERLAY_TTL_SECONDS");
        cfg_.FLOODFILL_OVERLAY_ALPHA = read_required<float>(*cfg, "FLOODFILL_OVERLAY_ALPHA");
        cfg_.MIN_AREA = read_required<int>(*cfg, "MIN_AREA");
        cfg_.MIN_WIDTH = read_required<int>(*cfg, "MIN_WIDTH");
        cfg_.MIN_HEIGHT = read_required<int>(*cfg, "MIN_HEIGHT");
        cfg_.MIN_CONTRAST = read_required<float>(*cfg, "MIN_CONTRAST");
        std::cout << "[STATIC] config: MAX_TARGETS=" << cfg_.MAX_TARGETS
                  << " CLICK_PADDING=" << cfg_.CLICK_PADDING
                  << " REMOVE_PADDING=" << cfg_.REMOVE_PADDING
                  << " FALLBACK_BOX_SIZE=" << cfg_.FALLBACK_BOX_SIZE
                  << " MAX_AREA_RATIO=" << cfg_.MAX_AREA_RATIO
                  << " CLICK_EQUALIZE=" << (cfg_.CLICK_EQUALIZE ? "true" : "false")
                  << " FLOODFILL_LO_DIFF=" << cfg_.FLOODFILL_LO_DIFF
                  << " FLOODFILL_HI_DIFF=" << cfg_.FLOODFILL_HI_DIFF
                  << " OVERLAY_TTL_SECONDS=" << cfg_.OVERLAY_TTL_SECONDS
                  << " MIN_AREA=" << cfg_.MIN_AREA
                  << " MIN_WIDTH=" << cfg_.MIN_WIDTH
                  << " MIN_HEIGHT=" << cfg_.MIN_HEIGHT
                  << " MIN_CONTRAST=" << cfg_.MIN_CONTRAST
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
    if (cfg_.CLICK_EQUALIZE) {
        cv::equalizeHist(search, search_for_fill);
    }

    cv::Mat flood_mask(search_for_fill.rows + 2, search_for_fill.cols + 2, CV_8UC1, cv::Scalar(0));
    cv::Rect bounding;
    const int flags = 4 | cv::FLOODFILL_MASK_ONLY | (255 << 8);
    const cv::Scalar lo(cfg_.FLOODFILL_LO_DIFF);
    const cv::Scalar hi(cfg_.FLOODFILL_HI_DIFF);

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

    if (roi.area() < static_cast<float>(cfg_.MIN_AREA) ||
        roi.width < cfg_.MIN_WIDTH ||
        roi.height < cfg_.MIN_HEIGHT) {
        const float half = static_cast<float>(cfg_.FALLBACK_BOX_SIZE) * 0.5f;
        roi = cv::Rect2f(static_cast<float>(x) - half, static_cast<float>(y) - half,
                         static_cast<float>(cfg_.FALLBACK_BOX_SIZE),
                         static_cast<float>(cfg_.FALLBACK_BOX_SIZE));
    }

    roi.x -= cfg_.CLICK_PADDING;
    roi.y -= cfg_.CLICK_PADDING;
    roi.width += cfg_.CLICK_PADDING * 2.0f;
    roi.height += cfg_.CLICK_PADDING * 2.0f;

    const float max_ratio = std::max(0.0f, std::min(cfg_.MAX_AREA_RATIO, 1.0f));
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
        const float half = static_cast<float>(cfg_.FALLBACK_BOX_SIZE) * 0.5f;
        roi = cv::Rect2f(static_cast<float>(x) - half, static_cast<float>(y) - half,
                         static_cast<float>(cfg_.FALLBACK_BOX_SIZE),
                         static_cast<float>(cfg_.FALLBACK_BOX_SIZE));
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
        if (point_in_rect_with_padding(it->bbox, x, y, cfg_.REMOVE_PADDING)) {
            const int removed_id = it->id;
            targets_.erase(it);
            if (log_cfg_.MANUAL_DETECTOR_LEVEL_LOGGER) {
                std::cout << "[STATIC] remove id=" << removed_id << std::endl;
            }
            return true;
        }
    }

    if (static_cast<int>(targets_.size()) >= cfg_.MAX_TARGETS) {
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
    if (contrast < cfg_.MIN_CONTRAST) {
        if (log_cfg_.MANUAL_DETECTOR_LEVEL_LOGGER) {
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
        overlay_expire_ms_ = now_ms + static_cast<long long>(cfg_.OVERLAY_TTL_SECONDS) * 1000;
    }

    if (log_cfg_.MANUAL_DETECTOR_LEVEL_LOGGER) {
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
        const double overlay_alpha = cfg_.FLOODFILL_OVERLAY_ALPHA;
        cv::addWeighted(frame, 1.0 - overlay_alpha, flood_fill_overlay_, overlay_alpha, 0.0, blended);
        blended.copyTo(frame, flood_fill_mask_);
    }
}
