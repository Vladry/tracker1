#include "tracker/tracker_manager.h"
#include <algorithm>
#include <cmath>

TrackerManager::TrackerManager(const Config& cfg)
        : cfg_(cfg) {}

void TrackerManager::reset() {
    tracks_.clear();
    targets_.clear();
    next_id_ = 1;
}

float TrackerManager::iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    cv::Rect2f inter = a & b;
    float ia = inter.area();
    if (ia <= 0.f) return 0.f;
    float ua = a.area() + b.area() - ia;
    return ua > 0.f ? ia / ua : 0.f;
}

cv::KalmanFilter TrackerManager::makeKalman(float cx, float cy) const {
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

    kf.statePost = (cv::Mat_<float>(4,1) << cx, cy, 0.f, 0.f);

    cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(cfg_.kalman_process_noise));
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(cfg_.kalman_meas_noise));
    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));

    return kf;
}

float TrackerManager::speedPx(const Track& t) const {
    if (!t.kf_initialized) return 0.f;
    float vx = t.kf.statePost.at<float>(2);
    float vy = t.kf.statePost.at<float>(3);
    return std::sqrt(vx*vx + vy*vy);
}

std::vector<Target> TrackerManager::update(
        const cv::Mat&,
        const std::vector<cv::Rect2f>& detections)
{
    const auto now = std::chrono::steady_clock::now();

    // 1. Predict (Kalman, но НЕ двигаем bbox если цель стоит)
    if (cfg_.use_kalman) {
        for (auto& t : tracks_) {
            if (!t.kf_initialized) continue;

            cv::Mat p = t.kf.predict();
            if (speedPx(t) >= cfg_.stationary_speed_px) {
                float px = p.at<float>(0);
                float py = p.at<float>(1);
                t.bbox.x = px - t.bbox.width  * 0.5f;
                t.bbox.y = py - t.bbox.height * 0.5f;
            } else {
                t.kf.statePost.at<float>(2) = 0.f;
                t.kf.statePost.at<float>(3) = 0.f;
                t.kf.statePre = t.kf.statePost.clone();
            }
        }
    }

    // 2. Ассоциация det -> track
    std::vector<int> det_to_track(detections.size(), -1);

    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
        float best_iou = 0.f;
        int best_di = -1;

        for (size_t di = 0; di < detections.size(); ++di) {
            if (det_to_track[di] != -1) continue;
            float v = iou(tracks_[ti].bbox, detections[di]);
            if (v > best_iou) {
                best_iou = v;
                best_di = (int)di;
            }
        }

        if (best_di >= 0 && best_iou >= cfg_.assoc_iou_threshold) {
            det_to_track[(size_t)best_di] = (int)ti;
        }
    }

    // 3. Обновление треков по детекциям
    for (size_t di = 0; di < detections.size(); ++di) {
        int ti = det_to_track[di];
        if (ti < 0) continue;

        auto& t = tracks_[(size_t)ti];
        const auto& d = detections[di];

        t.bbox = d;
        t.last_seen = now;
        t.hits++;

        float cx = d.x + d.width  * 0.5f;
        float cy = d.y + d.height * 0.5f;

        if (cfg_.use_kalman) {
            if (!t.kf_initialized) {
                t.kf = makeKalman(cx, cy);
                t.kf_initialized = true;
            } else {
                cv::Mat m = (cv::Mat_<float>(2,1) << cx, cy);
                t.kf.correct(m);

                float fx = t.kf.statePost.at<float>(0);
                float fy = t.kf.statePost.at<float>(1);
                t.bbox.x = fx - t.bbox.width  * 0.5f;
                t.bbox.y = fy - t.bbox.height * 0.5f;
            }

            if (speedPx(t) < cfg_.stationary_speed_px) {
                t.kf.statePost.at<float>(2) = 0.f;
                t.kf.statePost.at<float>(3) = 0.f;
                t.kf.statePre = t.kf.statePost.clone();
            }
        }
    }

    // 4. Spawn новых треков
    for (size_t di = 0; di < detections.size(); ++di) {
        if (det_to_track[di] != -1) continue;
        if ((int)tracks_.size() >= cfg_.max_targets) break;

        const auto& d = detections[di];
        bool block = false;

        for (const auto& t : tracks_) {
            double age = std::chrono::duration<double>(now - t.last_seen).count();
            if (age > cfg_.occlusion_timeout_sec * 0.5) continue;
            if (iou(t.bbox, d) >= cfg_.spawn_block_iou) {
                block = true;
                break;
            }
        }
        if (block) continue;

        Track t;
        t.id = next_id_++;
        t.bbox = d;
        t.hits = 1;
        t.confirmed = (t.hits >= cfg_.confirm_hits);
        t.last_seen = now;

        if (cfg_.use_kalman) {
            float cx = d.x + d.width  * 0.5f;
            float cy = d.y + d.height * 0.5f;
            t.kf = makeKalman(cx, cy);
            t.kf_initialized = true;
        }

        tracks_.push_back(std::move(t));
    }

    // 5. Удаление треков
    tracks_.erase(
            std::remove_if(tracks_.begin(), tracks_.end(),
                           [&](const Track& t){
                               double age = std::chrono::duration<double>(now - t.last_seen).count();
                               if (cfg_.use_kalman)
                                   return age > cfg_.occlusion_timeout_sec;
                               return age > cfg_.stationary_hold_sec;
                           }),
            tracks_.end()
    );

    rebuildTargets();
    return targets_;
}

void TrackerManager::rebuildTargets() {
    targets_.clear();
    for (const auto& t : tracks_) {
        Target tg;
        tg.id = t.id;
        tg.target_name = "T" + std::to_string(t.id);
        tg.bbox = t.bbox;
        tg.age_frames = t.hits;
        tg.missed_frames = 0;

        if (cfg_.use_kalman && t.kf_initialized) {
            tg.speedX_mps = t.kf.statePost.at<float>(2);
            tg.speedY_mps = t.kf.statePost.at<float>(3);
        } else {
            tg.speedX_mps = 0.f;
            tg.speedY_mps = 0.f;
        }

        targets_.push_back(std::move(tg));
    }
}

int TrackerManager::pickTargetId(int x, int y) const {
    for (const auto& t : targets_) {
        if (t.bbox.contains({(float)x, (float)y}))
            return t.id;
    }
    return -1;
}
