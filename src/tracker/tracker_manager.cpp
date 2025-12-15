#include "tracker/tracker_manager.h"
#include <algorithm>
#include <cmath>

cv::KalmanFilter TrackerManager::makeKF(float px, float py, float vx, float vy,
                                        float process_noise, float meas_noise) {
    cv::KalmanFilter kf(4, 2, 0, CV_32F);
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

static cv::Rect clampToFrame(const cv::Rect& r, int w, int h) {
    cv::Rect frame(0,0,w,h);
    return r & frame;
}

bool TrackerManager::extractAppearanceRef(const cv::Mat& frame_bgr,
                                          const cv::Rect2f& bbox,
                                          cv::Mat& out_ref) const {
    if (frame_bgr.empty()) return false;

    cv::Rect r((int)std::floor(bbox.x),
               (int)std::floor(bbox.y),
               (int)std::ceil(bbox.width),
               (int)std::ceil(bbox.height));

    r = clampToFrame(r, frame_bgr.cols, frame_bgr.rows);
    if (r.area() < cfg_.min_presence_area_px) return false;

    cv::Mat roi = frame_bgr(r);
    if (roi.empty()) return false;

    cv::Mat gray;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);

    cv::Mat small;
    cv::resize(gray, small, cv::Size(cfg_.appearance_patch_w, cfg_.appearance_patch_h), 0, 0, cv::INTER_AREA);

    out_ref = small;
    return !out_ref.empty();
}

bool TrackerManager::appearanceMatches(const cv::Mat& frame_bgr, const TrackKF& t) const {
    if (!t.appearance_valid) return false;

    cv::Mat cur;
    if (!extractAppearanceRef(frame_bgr, t.bbox, cur)) return false;

    // Mean L1 per pixel (0..255)
    double l1 = cv::norm(cur, t.appearance_ref, cv::NORM_L1);
    double mean_l1 = l1 / (double)cur.total();

    return (mean_l1 <= (double)cfg_.appearance_l1_thresh);
}

void TrackerManager::updateAppearanceRef(const cv::Mat& frame_bgr, TrackKF& t) const {
    cv::Mat cur;
    if (!extractAppearanceRef(frame_bgr, t.bbox, cur)) return;

    if (!t.appearance_valid) {
        t.appearance_ref = cur;
        t.appearance_valid = true;
        return;
    }

    // Exponential moving average in float, then back to 8-bit
    const float a = std::min(1.0f, std::max(0.0f, cfg_.appearance_update_alpha));
    cv::Mat ref_f, cur_f, out_f;
    t.appearance_ref.convertTo(ref_f, CV_32F);
    cur.convertTo(cur_f, CV_32F);
    out_f = (1.0f - a) * ref_f + a * cur_f;
    out_f.convertTo(t.appearance_ref, CV_8U);
}

std::vector<Target> TrackerManager::update(const cv::Mat& frame_bgr,
                                           const std::vector<cv::Rect2f>& detections) {
    // 1) Predict all tracks
    for (auto& t : tracks_) {
        cv::Mat p = t.kf.predict();
        float cx = p.at<float>(0);
        float cy = p.at<float>(1);

        // Limit prediction drift (pixels per frame)
        constexpr float MAX_PREDICT_STEP = 20.0f;

        float px = t.kf.statePost.at<float>(0);
        float py = t.kf.statePost.at<float>(1);

        float dx = cx - px;
        float dy = cy - py;

        float d = std::sqrt(dx*dx + dy*dy);
        if (d > MAX_PREDICT_STEP) {
            float s = MAX_PREDICT_STEP / d;
            cx = px + dx * s;
            cy = py + dy * s;
            t.kf.statePre.at<float>(0) = cx;
            t.kf.statePre.at<float>(1) = cy;
        }


        t.bbox.x = cx - t.bbox.width * 0.5f;
        t.bbox.y = cy - t.bbox.height * 0.5f;

        t.age++;
        t.missed++;
    }

    // 2) Associate detections -> tracks (greedy by IOU)
    std::vector<int> det_assigned(detections.size(), -1);

    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
        float best = 0.f;
        int best_di = -1;

        for (size_t di = 0; di < detections.size(); ++di) {
            if (det_assigned[di] != -1) continue;

            float v = iou(tracks_[ti].bbox, detections[di]);
            if (v > best) {
                best = v;
                best_di = (int)di;
            }
        }

        if (best_di != -1 && best >= cfg_.iou_threshold) {
            det_assigned[best_di] = (int)ti;

            const auto& d = detections[best_di];
            float mx = d.x + d.width * 0.5f;
            float my = d.y + d.height * 0.5f;

            cv::Mat m(2,1,CV_32F);
            m.at<float>(0) = mx;
            m.at<float>(1) = my;

            tracks_[ti].kf.correct(m);
            tracks_[ti].bbox = d;

            tracks_[ti].missed = 0;
            tracks_[ti].hits++;

            if (!tracks_[ti].confirmed && tracks_[ti].hits >= cfg_.confirm_hits) {
                tracks_[ti].confirmed = true;
                // Initialize appearance reference as soon as track becomes confirmed
                updateAppearanceRef(frame_bgr, tracks_[ti]);
            } else if (tracks_[ti].confirmed) {
                // keep adapting reference on real matches
                updateAppearanceRef(frame_bgr, tracks_[ti]);
            }
        }
    }

    // 3) Presence confirmation when detector is silent (critical fix)
    // If there are no detections, we try to confirm that confirmed tracks are still visible.
    if (detections.empty() && !frame_bgr.empty()) {
        for (auto& t : tracks_) {
            if (!t.confirmed) continue;

            if (appearanceMatches(frame_bgr, t)) {
                t.missed = 0;

                // CRITICAL: object is stationary -> kill velocity
                t.kf.statePost.at<float>(2) = 0.0f; // vx
                t.kf.statePost.at<float>(3) = 0.0f; // vy
                t.kf.statePre  = t.kf.statePost.clone();

                updateAppearanceRef(frame_bgr, t);
            }

        }
    }

    // 4) Spawn new tracks for unassigned detections (respect max_targets)
    for (size_t di = 0; di < detections.size(); ++di) {
        if (det_assigned[di] != -1) continue;
        if ((int)tracks_.size() >= cfg_.max_targets) break;

        const auto& d = detections[di];
        float mx = d.x + d.width * 0.5f;
        float my = d.y + d.height * 0.5f;

        TrackKF t;
        t.id = next_id_++;
        t.kf = makeKF(mx, my, 0.f, 0.f, cfg_.process_noise, cfg_.meas_noise);
        t.bbox = d;
        t.age = 0;
        t.missed = 0;
        t.hits = 1;
        t.confirmed = (t.hits >= cfg_.confirm_hits);

        // capture appearance early (even if not confirmed yet), harmless and helps fast lock
        updateAppearanceRef(frame_bgr, t);

        tracks_.push_back(std::move(t));
    }

    // 5) Prune tracks
    // - Unconfirmed: delete by max_missed_frames
    // - Confirmed: allow stationary_grace_frames more, because detector may output nothing
    tracks_.erase(
            std::remove_if(tracks_.begin(), tracks_.end(), [&](const TrackKF& t) {
                int limit = cfg_.max_missed_frames;
                if (t.confirmed) limit += cfg_.stationary_grace_frames;
                return t.missed > limit;
            }),
            tracks_.end()
    );

    // 6) Export targets
    rebuildTargets();
    return targets_;
}

void TrackerManager::rebuildTargets() {
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

        // Остальные поля Target остаются как у вас (overlay их использует)
        targets_.push_back(std::move(tg));
    }
}

int TrackerManager::pickTargetId(int x, int y) const {
    for (const auto& t : targets_) {
        if (t.bbox.contains(cv::Point2f((float)x, (float)y))) return t.id;
    }
    return -1;
}
