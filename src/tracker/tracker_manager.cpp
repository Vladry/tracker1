#include "tracker/tracker_manager.h"
#include <algorithm>

static cv::Rect clampRect(const cv::Rect& r, int w, int h) {
    return r & cv::Rect(0, 0, w, h);
}

TrackerManager::TrackerManager(const Config& cfg) : cfg_(cfg) {}

void TrackerManager::reset() {
    tracks_.clear();
    targets_.clear();
    next_id_ = 1;
}

float TrackerManager::iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    if (uni <= 0.0f) return 0.0f;
    return inter / uni;
}

bool TrackerManager::extractTemplate(const cv::Mat& frame_bgr,
                                     const cv::Rect2f& bbox,
                                     cv::Mat& out_tmpl) const
{
    if (frame_bgr.empty()) return false;

    cv::Rect r(
            (int)bbox.x,
            (int)bbox.y,
            (int)bbox.width,
            (int)bbox.height
    );

    r = clampRect(r, frame_bgr.cols, frame_bgr.rows);
    if (r.area() < cfg_.min_template_area_px) return false;

    cv::Mat roi = frame_bgr(r);
    if (roi.empty()) return false;

    cv::Mat gray;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, out_tmpl,
               cv::Size(cfg_.tmpl_patch_w, cfg_.tmpl_patch_h),
               0, 0, cv::INTER_AREA);

    return !out_tmpl.empty();
}

void TrackerManager::updateTemplate(Track& tr, const cv::Mat& new_tmpl) const {
    if (new_tmpl.empty()) return;

    if (tr.tmpl_gray.empty()) {
        tr.tmpl_gray = new_tmpl.clone();
        return;
    }

    float a = std::clamp(cfg_.tmpl_update_alpha, 0.0f, 1.0f);

    cv::Mat ref_f, cur_f;
    tr.tmpl_gray.convertTo(ref_f, CV_32F);
    new_tmpl.convertTo(cur_f, CV_32F);

    cv::Mat blended = (1.0f - a) * ref_f + a * cur_f;
    blended.convertTo(tr.tmpl_gray, CV_8U);
}

bool TrackerManager::templateTrackOne(const cv::Mat& frame_bgr, Track& tr) const {
    if (!cfg_.enable_template_tracking) return false;
    if (tr.tmpl_gray.empty()) return false;

    const cv::Rect2f& b = tr.bbox;

    cv::Rect search(
            (int)b.x - cfg_.tmpl_search_px,
            (int)b.y - cfg_.tmpl_search_px,
            (int)b.width  + 2 * cfg_.tmpl_search_px,
            (int)b.height + 2 * cfg_.tmpl_search_px
    );

    search = clampRect(search, frame_bgr.cols, frame_bgr.rows);
    if (search.width < cfg_.tmpl_patch_w || search.height < cfg_.tmpl_patch_h)
        return false;

    cv::Mat gray;
    cv::cvtColor(frame_bgr(search), gray, cv::COLOR_BGR2GRAY);

    cv::Mat result;
    cv::matchTemplate(gray, tr.tmpl_gray, result, cv::TM_CCOEFF_NORMED);

    double minVal, maxVal;
    cv::Point maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, nullptr, &maxLoc);

    if (maxVal < cfg_.tmpl_min_score) return false;

    tr.bbox.x = (float)(search.x + maxLoc.x);
    tr.bbox.y = (float)(search.y + maxLoc.y);

    cv::Mat new_tmpl;
    if (extractTemplate(frame_bgr, tr.bbox, new_tmpl)) {
        updateTemplate(tr, new_tmpl);
    }

    return true;
}

bool TrackerManager::overlapsExisting(const cv::Rect2f& det) const {
    for (const auto& tr : tracks_) {
        if (iou(tr.bbox, det) >= cfg_.seed_overlap_iou)
            return true;
    }
    return false;
}

std::vector<Target> TrackerManager::update(const cv::Mat& frame_bgr,
                                           const std::vector<cv::Rect2f>& new_detections)
{
    const auto now = std::chrono::steady_clock::now();

    // 1) Follow existing tracks
    for (auto& tr : tracks_) {
        if (templateTrackOne(frame_bgr, tr)) {
            tr.last_seen = now;
            tr.hits++;
            if (!tr.confirmed && tr.hits >= cfg_.confirm_hits)
                tr.confirmed = true;
        }
    }

    // 2) Seed new tracks (detector role)
    for (const auto& d : new_detections) {
        if ((int)tracks_.size() >= cfg_.max_targets) break;
        if (overlapsExisting(d)) continue;

        Track tr;
        tr.id = next_id_++;
        tr.bbox = d;
        tr.hits = 1;
        tr.confirmed = (tr.hits >= cfg_.confirm_hits);
        tr.last_seen = now;

        extractTemplate(frame_bgr, tr.bbox, tr.tmpl_gray);
        tracks_.push_back(std::move(tr));
    }

    // 3) Delete only by timeout
    const auto timeout = std::chrono::duration<double>(cfg_.occlusion_timeout_sec);
    tracks_.erase(
            std::remove_if(tracks_.begin(), tracks_.end(),
                           [&](const Track& tr){
                               return (now - tr.last_seen) > timeout;
                           }),
            tracks_.end()
    );

    // 4) Build targets
    rebuildTargets();
    return targets_;
}

void TrackerManager::rebuildTargets() {
    targets_.clear();
    for (const auto& tr : tracks_) {
        Target t;
        t.id = tr.id;
        t.target_name = "T" + std::to_string(tr.id);
        t.bbox = tr.bbox;
        t.age_frames = tr.hits;
        t.missed_frames = 0;
        t.speedX_mps = 0.0f;
        t.speedY_mps = 0.0f;
        targets_.push_back(std::move(t));
    }
}

int TrackerManager::pickTargetId(int x, int y) const {
    for (const auto& t : targets_) {
        if (t.bbox.contains({(float)x, (float)y}))
            return t.id;
    }
    return -1;
}
