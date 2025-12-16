#include "tracker/tracker_manager.h"

#include <algorithm>
#include <iostream>

TrackerManager::TrackerManager(const Config& cfg) : cfg_(cfg) {}

void TrackerManager::reset() {
    tracks_.clear();
    targets_.clear();
    next_id_ = 1;
    frame_index_ = 0;

}

float TrackerManager::iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    cv::Rect2f inter = a & b;
    float ia = inter.area();
    if (ia <= 0.f) return 0.f;
    float ua = a.area() + b.area() - ia;
    return ua > 0.f ? ia / ua : 0.f;
}

cv::Ptr<cv::Tracker> TrackerManager::createTracker() const {
#if CV_VERSION_MAJOR >= 4
    if (cfg_.use_csrt)
        return cv::TrackerCSRT::create();
    else
        return cv::TrackerKCF::create();
#else
    return cv::TrackerKCF::create();
#endif
}

void TrackerManager::initTracker(Track& t, const cv::Mat& frame, const cv::Rect2f& bbox) {
    t.tracker = createTracker();
    t.bbox = bbox;

    cv::Rect r(
            (int)bbox.x,
            (int)bbox.y,
            (int)bbox.width,
            (int)bbox.height
    );

    t.tracker->init(frame, r);
    t.tracker_ok = true;
    t.last_ok = Clock::now();
}

bool TrackerManager::spawnBlocked(const cv::Rect2f& det) const {
    for (const auto& t : tracks_) {
        if (iou(t.bbox, det) >= cfg_.spawn_block_iou)
            return true;
    }
    return false;
}

std::vector<Target> TrackerManager::update(const cv::Mat& frame,
                                           const std::vector<cv::Rect2f>& detections)
{
    frame_index_++;

    const auto now = Clock::now();

    // 1) Update visual trackers
    for (auto& t : tracks_) {
        if (!t.tracker) {
            t.tracker_ok = false;
            continue;
        }

        cv::Rect r(
                (int)t.bbox.x,
                (int)t.bbox.y,
                (int)t.bbox.width,
                (int)t.bbox.height
        );

// обновляем visual tracker НЕ каждый кадр
        static constexpr int TRACKER_UPDATE_EVERY = 30; // 1 = каждый кадр, 2 = через кадр, 3 = через два

        bool ok = true;

        if ((frame_index_ % TRACKER_UPDATE_EVERY) == 0) {
            ok = t.tracker->update(frame, r);
        }

        if (ok && r.width > 2 && r.height > 2) {
            t.tracker_ok = true;
            t.bbox = cv::Rect2f(
                    (float)r.x,
                    (float)r.y,
                    (float)r.width,
                    (float)r.height
            );
            t.last_ok = now;
        } else {
            t.tracker_ok = false;
        }

    }

    // 2) Optional resync with detections
    std::vector<int> det_used(detections.size(), 0);

    if (cfg_.allow_resync && cfg_.resync_every_n_frames > 0 &&
        (frame_index_ % (uint64_t)cfg_.resync_every_n_frames == 0))
    {
        for (size_t di = 0; di < detections.size(); ++di) {
            float best = 0.f;
            int best_ti = -1;

            for (size_t ti = 0; ti < tracks_.size(); ++ti) {
                float v = iou(tracks_[ti].bbox, detections[di]);
                if (v > best) {
                    best = v;
                    best_ti = (int)ti;
                }
            }

            if (best_ti >= 0 && best >= cfg_.det_match_iou) {
                initTracker(tracks_[(size_t)best_ti], frame, detections[di]);
                tracks_[(size_t)best_ti].hits++;
                det_used[di] = 1;
            }
        }
    }

    // 3) Spawn new tracks
    for (size_t di = 0; di < detections.size(); ++di) {
        if ((int)tracks_.size() >= cfg_.max_targets) break;
        if (det_used[di]) continue;
        if (spawnBlocked(detections[di])) continue;

        Track t;
        t.id = next_id_++;
        t.hits = 1;
        initTracker(t, frame, detections[di]);
        tracks_.push_back(std::move(t));
    }

    // 4) Prune lost tracks
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [&](const Track& t){
                                     double age = std::chrono::duration<double>(now - t.last_ok).count();
                                     return age > cfg_.occlusion_timeout_sec;
                                 }),
                  tracks_.end());

    // 5) Debug
    if (cfg_.debug_logs && cfg_.debug_every_n_frames > 0 &&
        (frame_index_ % (uint64_t)cfg_.debug_every_n_frames == 0))
    {
        int okc = 0;
        for (const auto& t : tracks_) if (t.tracker_ok) okc++;
/*        std::cout << "[tracker] frame=" << frame_index_
                  << " tracks=" << tracks_.size()
                  << " ok=" << okc
                  << " dets=" << detections.size()
                  << std::endl;*/
    }

    rebuildTargets();
    return targets_;
}

void TrackerManager::rebuildTargets() {
    targets_.clear();
    targets_.reserve(tracks_.size());

    for (const auto& tr : tracks_) {
        Target tg;
        tg.id = tr.id;
        tg.target_name = "T" + std::to_string(tr.id);
        tg.bbox = tr.bbox;
        tg.age_frames = tr.hits;
        tg.missed_frames = 0;
        tg.speedX_mps = 0.f;
        tg.speedY_mps = 0.f;
        targets_.push_back(std::move(tg));
    }
}

int TrackerManager::pickTargetId(int x, int y) const {
    for (const auto& t : targets_) {
        if (t.bbox.contains(cv::Point2f((float)x, (float)y)))
            return t.id;
    }
    return -1;
}

bool TrackerManager::hasTargetId(int id) const {
    for (const auto& t : targets_) {
        if (t.id == id) return true;
    }
    return false;
}
