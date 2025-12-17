#include "tracker_manager.h"
#include <algorithm>

TrackerManager::TrackerManager(const Config& cfg) : cfg_(cfg) {}

void TrackerManager::reset() {
    tracks_.clear();
    targets_.clear();
    next_id_ = 1;
}

float TrackerManager::iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    if (uni <= 0.f) return 0.f;
    return inter / uni;
}

std::vector<Target> TrackerManager::update(const std::vector<cv::Rect2f>& detections) {
    for (auto& t : tracks_) {
        t.age++;
        t.missed++;
    }

    std::vector<int> det_used(detections.size(), 0);

    for (auto& tr : tracks_) {
        float best = 0.f;
        int best_di = -1;

        for (size_t di = 0; di < detections.size(); ++di) {
            if (det_used[di]) continue;
            float v = iou(tr.bbox, detections[di]);
            if (v > best) { best = v; best_di = (int)di; }
        }

        if (best_di != -1 && best >= cfg_.iou_threshold) {
            tr.bbox = detections[(size_t)best_di];
            tr.missed = 0;
            det_used[(size_t)best_di] = 1;
        }
    }

    for (size_t di = 0; di < detections.size(); ++di) {
        if (det_used[di]) continue;
        if ((int)tracks_.size() >= cfg_.max_targets) break;

        Track t;
        t.id = next_id_++;
        t.bbox = detections[di];
        t.age = 0;
        t.missed = 0;
        tracks_.push_back(std::move(t));
    }

    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
        [&](const Track& t){ return t.missed > cfg_.max_missed_frames; }),
        tracks_.end());

    targets_.clear();
    targets_.reserve(tracks_.size());
    for (const auto& tr : tracks_) {
        Target tg;
        tg.id = tr.id;
        tg.target_name = "T" + std::to_string(tr.id);
        tg.bbox = tr.bbox;
        tg.age_frames = tr.age;
        tg.missed_frames = tr.missed;
        targets_.push_back(std::move(tg));
    }
    return targets_;
}

int TrackerManager::pickTargetId(int x, int y) const {
    for (const auto& t : targets_) {
        if (t.bbox.contains(cv::Point2f((float)x, (float)y))) return t.id;
    }
    return -1;
}

bool TrackerManager::hasTargetId(int id) const {
    for (const auto& t : targets_) if (t.id == id) return true;
    return false;
}
