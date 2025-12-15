#include "tracker/tracker_manager.h"
#include "util/rect_utils.h"
#include <algorithm>
#include <cmath>

TrackerManager::TrackerManager(const Config& cfg) : cfg_(cfg) {}

void TrackerManager::reset() {
    tracks_.clear();
    targets_.clear();
    next_id_ = 1;
}

cv::KalmanFilter TrackerManager::makeKF(float px, float py, float vx, float vy,
                                        float process_noise, float meas_noise)
{
    cv::KalmanFilter kf(4, 2, 0, CV_32F);

    // state: [x, y, vx, vy]
    kf.transitionMatrix = (cv::Mat_<float>(4,4) <<
        1,0,1,0,
        0,1,0,1,
        0,0,1,0,
        0,0,0,1
    );
    kf.measurementMatrix = (cv::Mat_<float>(2,4) <<
        1,0,0,0,
        0,1,0,0
    );

    kf.statePre.at<float>(0) = px;
    kf.statePre.at<float>(1) = py;
    kf.statePre.at<float>(2) = vx;
    kf.statePre.at<float>(3) = vy;
    kf.statePost = kf.statePre.clone();

    cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(process_noise));
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(meas_noise));
    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));

    return kf;
}

float TrackerManager::trackSpeedPxPerFrame(const TrackKF& t) {
    float vx = t.kf.statePost.at<float>(2);
    float vy = t.kf.statePost.at<float>(3);
    return std::sqrt(vx*vx + vy*vy);
}

void TrackerManager::associateGreedy(const std::vector<cv::Rect2f>& detections,
                                     std::vector<int>& det_to_track)
{
    det_to_track.assign(detections.size(), -1);
    if (detections.empty() || tracks_.empty()) return;

    std::vector<bool> track_used(tracks_.size(), false);

    // For each track, pick best detection (greedy).
    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
        float best_score = -1.0f;
        int best_di = -1;

        for (size_t di = 0; di < detections.size(); ++di) {
            if (det_to_track[di] != -1) continue;

            float iou_v = util::iou(tracks_[ti].bbox, detections[di]);
            float dist  = util::centerDistance(tracks_[ti].bbox, detections[di]);

            bool pass_gate = (iou_v >= cfg_.match_iou_threshold) ||
                             ((iou_v >= cfg_.assoc_iou_min) && (dist <= cfg_.assoc_dist_max_px));

            if (!pass_gate) continue;

            float dist_score = 0.0f;
            if (cfg_.assoc_dist_max_px > 1e-3f) {
                dist_score = 1.0f - std::min(1.0f, dist / cfg_.assoc_dist_max_px);
            }
            float score = cfg_.score_iou_w * iou_v + cfg_.score_dist_w * dist_score;

            if (score > best_score) {
                best_score = score;
                best_di = (int)di;
            }
        }

        if (best_di != -1) {
            det_to_track[best_di] = (int)ti;
            track_used[ti] = true;
        }
    }
}

void TrackerManager::spawnNewTracks(const std::vector<cv::Rect2f>& detections,
                                    const std::vector<int>& det_to_track)
{
    // spawn from unmatched detections, prefer larger boxes (already sorted in main after merge)
    for (size_t di = 0; di < detections.size(); ++di) {
        if (det_to_track[di] != -1) continue;
        if ((int)tracks_.size() >= cfg_.max_targets) break;

        const auto& d = detections[di];
        float mx = d.x + d.width * 0.5f;
        float my = d.y + d.height * 0.5f;

        TrackKF t;
        t.id = next_id_++;
        t.kf = makeKF(mx, my, 0.0f, 0.0f, cfg_.process_noise, cfg_.meas_noise);
        t.bbox = d;
        t.age = 0;
        t.missed = 0;
        t.hits = 1;
        t.confirmed = (t.hits >= cfg_.confirm_hits);

        tracks_.push_back(std::move(t));
    }
}

bool TrackerManager::shouldDelete(const TrackKF& t) const {
    // Unconfirmed tracks are more aggressively deleted (reduce clutter).
    int base = cfg_.max_missed_frames;
    int allowed = base;

    if (!t.confirmed) {
        allowed = std::max(2, base / 3);
        return t.missed > allowed;
    }

    // Confirmed tracks: if target is stationary (KF velocity small),
    // allow longer grace time to prevent disappearance when motion-detector stops producing detections.
    float sp = trackSpeedPxPerFrame(t);
    if (sp <= cfg_.stationary_speed_thresh) {
        allowed = std::max(base, cfg_.stationary_grace_frames);
    }

    return t.missed > allowed;
}

std::vector<Target> TrackerManager::update(const std::vector<cv::Rect2f>& detections)
{
    // 1) Predict
    for (auto& t : tracks_) {
        cv::Mat p = t.kf.predict();
        float cx = p.at<float>(0);
        float cy = p.at<float>(1);

        // keep last known size; update position from predicted center
        t.bbox.x = cx - t.bbox.width  * 0.5f;
        t.bbox.y = cy - t.bbox.height * 0.5f;

        t.age++;
        t.missed++;
    }

    // 2) Associate (greedy)
    std::vector<int> det_to_track;
    associateGreedy(detections, det_to_track);

    // 3) Correct matches
    for (size_t di = 0; di < detections.size(); ++di) {
        int ti = det_to_track[di];
        if (ti < 0 || ti >= (int)tracks_.size()) continue;

        auto& tr = tracks_[(size_t)ti];
        const auto& d = detections[di];

        float mx = d.x + d.width * 0.5f;
        float my = d.y + d.height * 0.5f;

        cv::Mat m(2,1,CV_32F);
        m.at<float>(0) = mx;
        m.at<float>(1) = my;

        tr.kf.correct(m);
        tr.bbox = d;
        tr.missed = 0;
        tr.hits++;
        if (!tr.confirmed && tr.hits >= cfg_.confirm_hits) tr.confirmed = true;
    }

    // 4) Spawn new tracks for unmatched detections
    spawnNewTracks(detections, det_to_track);

    // 5) Enforce capacity if overshoot (shouldn't happen often, but keep deterministic)
    if ((int)tracks_.size() > cfg_.max_targets) {
        // keep confirmed first, then by hits, then by lowest missed
        std::sort(tracks_.begin(), tracks_.end(), [](const TrackKF& a, const TrackKF& b){
            if (a.confirmed != b.confirmed) return a.confirmed > b.confirmed;
            if (a.hits != b.hits) return a.hits > b.hits;
            return a.missed < b.missed;
        });
        tracks_.resize((size_t)cfg_.max_targets);
    }

    // 6) Prune deleted tracks
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
        [&](const TrackKF& t){ return shouldDelete(t); }),
        tracks_.end());

    // 7) Build output targets
    targets_.clear();
    targets_.reserve(tracks_.size());

    for (auto& tr : tracks_) {
        Target tg;
        tg.id = tr.id;
        tg.target_name = "T" + std::to_string(tr.id);
        tg.bbox = tr.bbox;
        tg.age_frames = tr.age;
        tg.missed_frames = tr.missed;
        tg.speedX_mps = tr.kf.statePost.at<float>(2);
        tg.speedY_mps = tr.kf.statePost.at<float>(3);
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
