#include "tracker_manager.h"
#include <algorithm>

static cv::KalmanFilter makeKF(float px, float py, float vx, float vy, float process_noise, float meas_noise) {
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

std::vector<Target> TrackerManager::update(const std::vector<cv::Rect2f>& detections) {
    for (auto& t : tracks_) {
        cv::Mat p = t.kf.predict();
        float cx = p.at<float>(0);
        float cy = p.at<float>(1);
        t.bbox.x = cx - t.bbox.width * 0.5f;
        t.bbox.y = cy - t.bbox.height * 0.5f;
        t.age++;
        t.missed++;
    }

    std::vector<int> det_assigned(detections.size(), -1);

    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
        float best = 0.f;
        int best_di = -1;
        for (size_t di = 0; di < detections.size(); ++di) {
            if (det_assigned[di] != -1) continue;
            float v = iou(tracks_[ti].bbox, detections[di]);
            if (v > best) { best = v; best_di = (int)di; }
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
        }
    }

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
        tracks_.push_back(std::move(t));
    }

    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
        [&](const TrackKF& t){ return t.missed > cfg_.max_missed_frames; }),
        tracks_.end());

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
