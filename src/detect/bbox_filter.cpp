#include "detect/bbox_filter.h"

#include <algorithm>

namespace detect {

    BBoxFilter::BBoxFilter(Config cfg) : cfg_(cfg) {}

    float BBoxFilter::iou(const cv::Rect2f& a, const cv::Rect2f& b) {
        const float x1 = std::max(a.x, b.x);
        const float y1 = std::max(a.y, b.y);
        const float x2 = std::min(a.x + a.width,  b.x + b.width);
        const float y2 = std::min(a.y + a.height, b.y + b.height);

        const float iw = x2 - x1;
        const float ih = y2 - y1;
        if (iw <= 0.f || ih <= 0.f) return 0.f;

        const float inter = iw * ih;
        const float uni = a.area() + b.area() - inter;
        return (uni > 0.f) ? (inter / uni) : 0.f;
    }

    bool BBoxFilter::sanityOk(const cv::Rect2f& r,
                              const cv::Size& frame_size) const {
        if (r.width <= 0.f || r.height <= 0.f) return false;

        if (r.width  < cfg_.min_w || r.height < cfg_.min_h) return false;
        if (r.width  > cfg_.max_w || r.height > cfg_.max_h) return false;

        const float ar = r.width / r.height;
        if (ar < cfg_.min_ar || ar > cfg_.max_ar) return false;

        const float frame_area =
                static_cast<float>(frame_size.width * frame_size.height);
        const float max_area =
                std::min(cfg_.max_area_px, cfg_.max_area_frac * frame_area);

        const float a = r.area();
        if (a < cfg_.min_area_px) return false;
        if (a > max_area)        return false;

        // ensure inside frame
        if (r.x < 0.f || r.y < 0.f) return false;
        if (r.x + r.width  > frame_size.width)  return false;
        if (r.y + r.height > frame_size.height) return false;

        return true;
    }

    bool BBoxFilter::areaJumpOk(const cv::Rect2f& prev,
                                const cv::Rect2f& cur) const {
        const float a0 = prev.area();
        const float a1 = cur.area();
        if (a0 <= 0.f || a1 <= 0.f) return false;

        const float ratio = (a1 > a0) ? (a1 / a0) : (a0 / a1);
        return ratio <= cfg_.max_area_jump_ratio;
    }

    std::vector<cv::Rect2f>
    BBoxFilter::process(const std::vector<cv::Rect2f>& dets,
                        const cv::Size& frame_size) {

        std::vector<cv::Rect2f> clean;
        clean.reserve(dets.size());

        for (const auto& r : dets) {
            if (sanityOk(r, frame_size))
                clean.push_back(r);
        }

        std::vector<bool> used_prev(prev_.size(), false);
        std::vector<cv::Rect2f> out;
        out.reserve(clean.size() + prev_.size());

        for (const auto& cur : clean) {
            int best_i = -1;
            float best_iou = 0.f;

            for (int i = 0; i < static_cast<int>(prev_.size()); ++i) {
                if (used_prev[i]) continue;
                const float v = iou(prev_[i].r, cur);
                if (v > best_iou) {
                    best_iou = v;
                    best_i = i;
                }
            }

            if (best_i >= 0 && best_iou >= cfg_.match_iou) {
                if (areaJumpOk(prev_[best_i].r, cur)) {
                    out.push_back(cur);
                } else {
                    out.push_back(prev_[best_i].r);
                }
                used_prev[best_i] = true;
            } else {
                out.push_back(cur);
            }
        }

        for (int i = 0; i < static_cast<int>(prev_.size()); ++i) {
            if (used_prev[i]) continue;
            if (prev_[i].missed + 1 <= cfg_.hold_missed_frames) {
                prev_[i].missed += 1;
                out.push_back(prev_[i].r);
            }
        }

        prev_.clear();
        prev_.reserve(out.size());
        for (const auto& r : out) {
            prev_.push_back(TrackedDet{r, 0});
        }

        return out;
    }

} // namespace detect
