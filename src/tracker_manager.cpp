#include "tracker_manager.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>


TrackerManager::TrackerManager(const toml::table& tbl) {
    load_tracker_config(tbl);
}

bool TrackerManager::load_tracker_config(const toml::table& tbl) {
// ---------------------------- [tracker] ---------------------------
    try {
        const auto *tracker = tbl["tracker"].as_table();
        if (!tracker) {
            throw std::runtime_error("missing [tracker] table");
        }
        cfg_.iou_threshold = read_required<float>(*tracker, "iou_th");
        cfg_.max_missed_frames = read_required<int>(*tracker, "max_missed_frames");
        cfg_.max_targets = read_required<int>(*tracker, "max_targets");
        cfg_.leading_only = read_required<bool>(*tracker, "leading_only");
        cfg_.leading_min_speed = read_required<float>(*tracker, "leading_min_speed");
        cfg_.use_kalman = read_required<bool>(*tracker, "use_kalman");
        cfg_.kalman_process_noise = read_required<float>(*tracker, "kalman_process_noise");
        cfg_.kalman_measurement_noise = read_required<float>(*tracker, "kalman_measurement_noise");
        std::cout << "[TRK] config: iou_th=" << cfg_.iou_threshold
                  << " max_missed=" << cfg_.max_missed_frames
                  << " max_targets=" << cfg_.max_targets
                  << " leading_only=" << (cfg_.leading_only ? "true" : "false")
                  << " leading_min_speed=" << cfg_.leading_min_speed
                  << " use_kalman=" << (cfg_.use_kalman ? "true" : "false")
                  << " kalman_process_noise=" << cfg_.kalman_process_noise
                  << " kalman_measurement_noise=" << cfg_.kalman_measurement_noise
                  << std::endl;
        return true;

    } catch (const std::exception &e) {
        std::cerr << "tracker config load failed  " << e.what() << std::endl;
        return false;
    }
};


void TrackerManager::reset() {
    tracks_.clear();
    targets_.clear();
    next_id_ = 1;
    leading_id_ = -1;
}

float TrackerManager::iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    if (uni <= 0.f) return 0.f;
    return inter / uni;
}

void TrackerManager::init_kalman(Track &track, const cv::Point2f &center) const {
    track.kf = cv::KalmanFilter(4, 2, 0, CV_32F);
    track.kf.transitionMatrix = (cv::Mat_<float>(4, 4) <<
                                                       1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1);
    track.kf.measurementMatrix = (cv::Mat_<float>(2, 4) <<
                                                        1, 0, 0, 0,
            0, 1, 0, 0);
    track.kf.processNoiseCov = cv::Mat::eye(4, 4, CV_32F) * cfg_.kalman_process_noise;
    track.kf.measurementNoiseCov = cv::Mat::eye(2, 2, CV_32F) * cfg_.kalman_measurement_noise;
    track.kf.errorCovPost = cv::Mat::eye(4, 4, CV_32F);
    track.kf.statePost = (cv::Mat_<float>(4, 1) << center.x, center.y, 0.0f, 0.0f);
    track.kf_ready = true;
}

void TrackerManager::predict_kalman(Track &track) {
    if (!cfg_.use_kalman || !track.kf_ready) {
        return;
    }
    cv::Mat prediction = track.kf.predict();
    const float px = prediction.at<float>(0);
    const float py = prediction.at<float>(1);
    cv::Point2f center(px, py);
    if (track.has_center) {
        track.velocity = center - track.last_center;
    }
    track.last_center = center;
    track.has_center = true;
}

void TrackerManager::correct_kalman(Track &track, const cv::Point2f &center) {
    if (!cfg_.use_kalman || !track.kf_ready) {
        return;
    }
    cv::Mat measurement = (cv::Mat_<float>(2, 1) << center.x, center.y);
    cv::Mat estimate = track.kf.correct(measurement);
    track.last_center = cv::Point2f(estimate.at<float>(0), estimate.at<float>(1));
    track.has_center = true;
}

std::vector<Target> TrackerManager::update(const std::vector<cv::Rect2f>& detections) {
    std::cout << "[TRK] update: detections=" << detections.size()
              << " tracks_before=" << tracks_.size()
              << std::endl;
    for (auto& t : tracks_) {
        t.age++;
        t.missed++;
        predict_kalman(t);
        if (t.has_center) {
            t.bbox.x = t.last_center.x - t.bbox.width * 0.5f;
            t.bbox.y = t.last_center.y - t.bbox.height * 0.5f;
        }
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
            cv::Point2f center(tr.bbox.x + tr.bbox.width * 0.5f,
                               tr.bbox.y + tr.bbox.height * 0.5f);
            const cv::Point2f prev_center = tr.last_center;
            if (!tr.kf_ready) {
                init_kalman(tr, center);
            } else {
                correct_kalman(tr, center);
            }
            if (tr.has_center) {
                tr.velocity = center - prev_center;
            } else {
                tr.velocity = cv::Point2f(0.0f, 0.0f);
            }
            tr.last_center = center;
            tr.has_center = true;
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
        t.last_center = cv::Point2f(t.bbox.x + t.bbox.width * 0.5f,
                                    t.bbox.y + t.bbox.height * 0.5f);
        t.velocity = cv::Point2f(0.0f, 0.0f);
        t.has_center = true;
        init_kalman(t, t.last_center);
        tracks_.push_back(std::move(t));
    }

    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [&](const Track& t){ return t.missed > cfg_.max_missed_frames; }),
                  tracks_.end());
    std::cout << "[TRK] tracks_after_prune=" << tracks_.size() << std::endl;

    if (cfg_.leading_only && !tracks_.empty()) {
        cv::Point2f dir_sum(0.0f, 0.0f);
        float weight_sum = 0.0f;
        for (const auto& tr : tracks_) {
            if (tr.missed > 0) continue;
            float speed = std::sqrt(tr.velocity.x * tr.velocity.x +
                                    tr.velocity.y * tr.velocity.y);
            if (speed < cfg_.leading_min_speed) continue;
            cv::Point2f dir = tr.velocity * (1.0f / speed);
            dir_sum += dir * speed;
            weight_sum += speed;
        }

        if (weight_sum > 0.0f) {
            cv::Point2f dir = dir_sum * (1.0f / weight_sum);
            float dir_norm = std::sqrt(dir.x * dir.x + dir.y * dir.y);
            if (dir_norm > 1e-3f) {
                dir *= (1.0f / dir_norm);
                int best_index = -1;
                float best_proj = -std::numeric_limits<float>::infinity();
                for (size_t i = 0; i < tracks_.size(); ++i) {
                    if (tracks_[i].missed > 0) continue;
                    cv::Point2f center(tracks_[i].bbox.x + tracks_[i].bbox.width * 0.5f,
                                       tracks_[i].bbox.y + tracks_[i].bbox.height * 0.5f);
                    float proj = center.x * dir.x + center.y * dir.y;
                    if (proj > best_proj) {
                        best_proj = proj;
                        best_index = static_cast<int>(i);
                    }
                }

                if (best_index >= 0) {
                    leading_id_ = tracks_[static_cast<size_t>(best_index)].id;
                }
            }
        }

        if (leading_id_ >= 0) {
            bool leading_found = false;
            for (const auto& tr : tracks_) {
                if (tr.id == leading_id_) {
                    leading_found = true;
                    break;
                }
            }
            if (!leading_found) {
                leading_id_ = -1;
            }
        }

        if (leading_id_ < 0) {
            int best_index = -1;
            float best_area = -1.0f;
            for (size_t i = 0; i < tracks_.size(); ++i) {
                if (tracks_[i].missed > 0) continue;
                float area = tracks_[i].bbox.area();
                if (area > best_area) {
                    best_area = area;
                    best_index = static_cast<int>(i);
                }
            }
            if (best_index >= 0) {
                leading_id_ = tracks_[static_cast<size_t>(best_index)].id;
            }
        }
    }

    targets_.clear();
    if (cfg_.leading_only && leading_id_ >= 0) {
        targets_.reserve(1);
        for (const auto& tr : tracks_) {
            if (tr.id != leading_id_) continue;
            Target tg;
            tg.id = tr.id;
            tg.target_name = "T" + std::to_string(tr.id);
            tg.bbox = tr.bbox;
            tg.age_frames = tr.age;
            tg.missed_frames = tr.missed;
            targets_.push_back(std::move(tg));
            break;
        }
    } else {
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
    }
    std::cout << "[TRK] targets=" << targets_.size()
              << " leading_id=" << leading_id_
              << std::endl;
    return targets_;
}
