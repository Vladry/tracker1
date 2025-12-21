#include "overlay/overlay_renderer.h"

#include <algorithm>
#include <cstdio>
#include <deque>
#include <unordered_map>
#include <unordered_set>

//==============================================================================
// Dynamic bbox visual smoothing
//
// IMPORTANT:
//  - This affects ONLY rendering.
//  - Tracker / detector logic remains untouched.
//  - Each dynamic Target.id has its own smoothing history.
//==============================================================================

// Number of last frames used for smoothing dynamic bounding boxes.
// Larger value -> smoother, but more visual latency.
static constexpr int DYN_SMOOTH_WINDOW = 5;

// Per-target smoothing state (internal to this translation unit)
struct smooth_rect_state {
    std::deque<cv::Rect2f> history;
};

// Internal storage for smoothing (key = Target.id)
static std::unordered_map<int, smooth_rect_state> g_dyn_smooth;

//------------------------------------------------------------------------------
// Smooth bbox for rendering (moving average over last N frames)
//------------------------------------------------------------------------------
static cv::Rect smooth_bbox_for_render(int id, const cv::Rect& current) {
    auto& st = g_dyn_smooth[id];
    st.history.emplace_back(cv::Rect2f(current));

    if ((int)st.history.size() > DYN_SMOOTH_WINDOW)
        st.history.pop_front();

    cv::Rect2f acc(0, 0, 0, 0);
    for (const auto& r : st.history) {
        acc.x += r.x;
        acc.y += r.y;
        acc.width  += r.width;
        acc.height += r.height;
    }

    float inv = 1.0f / (float)st.history.size();
    acc.x      *= inv;
    acc.y      *= inv;
    acc.width  *= inv;
    acc.height *= inv;

    int x = (int)std::lround(acc.x);
    int y = (int)std::lround(acc.y);
    int w = (int)std::lround(acc.width);
    int h = (int)std::lround(acc.height);

    if (w < 1) w = 1;
    if (h < 1) h = 1;

    return cv::Rect(x, y, w, h);
}

//------------------------------------------------------------------------------
// ctor
//------------------------------------------------------------------------------
OverlayRenderer::OverlayRenderer(const Config& cfg)
        : cfg_(cfg) {}

//------------------------------------------------------------------------------
// Utility helpers
//------------------------------------------------------------------------------
cv::Rect OverlayRenderer::clip_rect(const cv::Rect& r, int w, int h) {
    int x1 = std::max(0, r.x);
    int y1 = std::max(0, r.y);
    int x2 = std::min(w, r.x + r.width);
    int y2 = std::min(h, r.y + r.height);

    if (x2 <= x1 || y2 <= y1)
        return cv::Rect(0, 0, 0, 0);

    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

void OverlayRenderer::draw_rect_alpha(
        cv::Mat& frame,
        const cv::Rect& r,
        const cv::Scalar& color,
        float alpha
) {
    if (alpha <= 0.0f)
        return;

    alpha = std::min(1.0f, alpha);

    cv::Rect rr = clip_rect(r, frame.cols, frame.rows);
    if (rr.width <= 0 || rr.height <= 0)
        return;

    cv::Mat roi = frame(rr);
    cv::Mat overlay(roi.size(), roi.type(), color);
    cv::addWeighted(overlay, alpha, roi, 1.0f - alpha, 0.0, roi);
}

void OverlayRenderer::draw_label(
        cv::Mat& frame,
        const cv::Point& org,
        const std::string& text,
        const cv::Scalar& color
) {
    int baseline = 0;
    cv::Size ts = cv::getTextSize(
            text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

    cv::Rect bg(
            org.x,
            org.y - ts.height - baseline - 4,
            ts.width + 6,
            ts.height + baseline + 6
    );

    bg = clip_rect(bg, frame.cols, frame.rows);
    if (bg.width > 0 && bg.height > 0)
        draw_rect_alpha(frame, bg, cv::Scalar(0, 0, 0), 0.35f);

    cv::putText(
            frame, text, org,
            cv::FONT_HERSHEY_SIMPLEX, 0.5,
            color, 1, cv::LINE_AA
    );
}

//------------------------------------------------------------------------------
// Render dynamic boxes (tracker targets)
//------------------------------------------------------------------------------
void OverlayRenderer::render(
        cv::Mat& frame,
        const std::vector<Target>& targets,
        int /*selected_id*/
) {
    // Fade background slightly if requested
    if (cfg_.hud_alpha > 0.0f) {
        cv::Rect hud(0, 0, frame.cols, 24);
        draw_rect_alpha(frame, hud, cv::Scalar(0, 0, 0), cfg_.hud_alpha);
    }

    // Track which ids are present in this frame (for cleanup)
    std::unordered_set<int> seen_ids;
    seen_ids.reserve(targets.size());

    for (const auto& t : targets) {
        seen_ids.insert(t.id);

        // IMPORTANT:
        // Raw bbox comes from tracker.
        // We apply smoothing ONLY for rendering.
        const cv::Rect r = smooth_bbox_for_render(t.id, t.bbox);

        // цвет динамических bbox-ов (плавно сглаженные)
        cv::rectangle(frame, r, cv::Scalar(0, 255, 0), 1); // green

#ifdef SHOW_DYNAMIC_IDS
        char buf[64];
        std::snprintf(buf, sizeof(buf), "dyn id=%d", t.id);
        draw_label(frame, cv::Point(r.x + 2, std::max(14, r.y - 2)),
                   buf, cv::Scalar(0, 0, 255));
#endif
    }

    // Cleanup smoothing states for disappeared targets
    for (auto it = g_dyn_smooth.begin(); it != g_dyn_smooth.end(); ) {
        if (seen_ids.find(it->first) == seen_ids.end())
            it = g_dyn_smooth.erase(it);
        else
            ++it;
    }
}

//------------------------------------------------------------------------------
// Render static (user-locked) boxes
//------------------------------------------------------------------------------
void OverlayRenderer::render_static_boxes(
        cv::Mat& frame,
        const std::vector<static_box>& boxes
) {
    for (const auto& sb : boxes) {
        const cv::Rect r = sb.rect;

        cv::Scalar color;
        switch (sb.state) {
            case static_box_state::attached:
                // цвет статических bbox-ов (выбрана цель!)
                color = cv::Scalar(0, 0, 255);      // red
                break;
            case static_box_state::pending_rebind:
                color = cv::Scalar(0, 255, 255);    // yellow
                break;
            case static_box_state::lost:
            default:
                color = cv::Scalar(128, 128, 128);  // gray
                break;
        }

        cv::rectangle(frame, r, color, 3);

#ifdef SHOW_IDS
        char buf[64];
        std::snprintf(buf, sizeof(buf), "static id=%d", sb.id);
        draw_label(frame, cv::Point(r.x + 2, std::max(14, r.y - 2)),
                   buf, color);
#endif
    }
}
