#include "tracks_manager.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

/*
  менеджер треков, отвечает за треки (IOU‑ассоциация, missed‑frames)
  -- см. описание в word-файле
 */

TracksManager::TracksManager(const toml::table& tbl) {
    load_tracker_config(tbl);
}

bool TracksManager::load_tracker_config(const toml::table& tbl) {
// ---------------------------- [tracker] ---------------------------
    try {
        const auto *tracker = tbl["tracker"].as_table();
        if (!tracker) {
            throw std::runtime_error("missing [tracker] table");
        }
        cfg_.IOU_THRESHOLD = read_required<float>(*tracker, "IOU_THRESHOLD");
        cfg_.MAX_MISSED_FRAMES = read_required<int>(*tracker, "MAX_MISSED_FRAMES");
        cfg_.MAX_TARGETS = read_required<int>(*tracker, "MAX_TARGETS");
        cfg_.LEADING_ONLY = read_required<bool>(*tracker, "LEADING_ONLY");
        cfg_.LEADING_MIN_SPEED = read_required<float>(*tracker, "LEADING_MIN_SPEED");
        std::cout << "[TRK] config: IOU_THRESHOLD=" << cfg_.IOU_THRESHOLD
                  << " MAX_MISSED_FRAMES=" << cfg_.MAX_MISSED_FRAMES
                  << " MAX_TARGETS=" << cfg_.MAX_TARGETS
                  << " LEADING_ONLY=" << (cfg_.LEADING_ONLY ? "true" : "false")
                  << " LEADING_MIN_SPEED=" << cfg_.LEADING_MIN_SPEED
                  << std::endl;
        return true;

    } catch (const std::exception &e) {
        std::cerr << "tracker config load failed  " << e.what() << std::endl;
        return false;
    }
};


void TracksManager::reset() {
    tracks_.clear();
    targets_.clear();
    next_id_ = 1;
    leading_id_ = -1;
}

float TracksManager::iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    if (uni <= 0.f) return 0.f;
    return inter / uni;
}



std::vector<Target> TracksManager::update(const std::vector<cv::Rect2f>& detections) {
    std::cout << "[TRK] update: detections=" << detections.size()
              << " tracks_before=" << tracks_.size()
              << std::endl;
    for (auto& t : tracks_) {
        t.missed++;
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

        if (best_di != -1 && best >= cfg_.IOU_THRESHOLD) {
            tr.bbox = detections[(size_t)best_di];
            tr.missed = 0;
            det_used[(size_t)best_di] = 1;
            cv::Point2f center(tr.bbox.x + tr.bbox.width * 0.5f,
                               tr.bbox.y + tr.bbox.height * 0.5f);
            const cv::Point2f prev_center = tr.last_center;

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
        if ((int)tracks_.size() >= cfg_.MAX_TARGETS) break;

        Track t;
        t.id = next_id_++;
        t.bbox = detections[di];
        t.missed = 0;
        t.last_center = cv::Point2f(t.bbox.x + t.bbox.width * 0.5f,
                                    t.bbox.y + t.bbox.height * 0.5f);
        t.velocity = cv::Point2f(0.0f, 0.0f);
        t.has_center = true;
        tracks_.push_back(std::move(t));
    }

    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [&](const Track& t){ return t.missed > cfg_.MAX_MISSED_FRAMES; }),
                  tracks_.end());
    std::cout << "[TRK] tracks_after_prune=" << tracks_.size() << std::endl;

    if (cfg_.LEADING_ONLY && !tracks_.empty()) {
        cv::Point2f dir_sum(0.0f, 0.0f);
        float weight_sum = 0.0f;
        for (const auto& tr : tracks_) {
            if (tr.missed > 0) continue;
            float speed = std::sqrt(tr.velocity.x * tr.velocity.x +
                                    tr.velocity.y * tr.velocity.y);
            if (speed < cfg_.LEADING_MIN_SPEED) continue;
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
    if (cfg_.LEADING_ONLY && leading_id_ >= 0) {
        targets_.reserve(1);
        for (const auto& tr : tracks_) {
            if (tr.id != leading_id_) continue;
            Target tg;
            tg.id = tr.id;
            tg.target_name = "T" + std::to_string(tr.id);
            tg.bbox = tr.bbox;
            tg.has_cross = false;
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
            tg.has_cross = false;
            tg.missed_frames = tr.missed;
            targets_.push_back(std::move(tg));
        }
    }
    std::cout << "[TRK] targets=" << targets_.size()
              << " leading_id=" << leading_id_
              << std::endl;
    return targets_;
}
