#include "static_box_manager.h"
#include <cmath>
#include <limits>

static float center_dist(const cv::Rect2f &a, const cv::Rect2f &b) {
    cv::Point2f ca(a.x + a.width * 0.5f, a.y + a.height * 0.5f);
    cv::Point2f cb(b.x + b.width * 0.5f, b.y + b.height * 0.5f);
    float dx = ca.x - cb.x;
    float dy = ca.y - cb.y;
    return std::sqrt(dx * dx + dy * dy);
}

StaticBoxManager::StaticBoxManager(const toml::table &tbl) {
    load_static_rebind_config(tbl);
}

void StaticBoxManager::update_trajectories(
        const std::vector<cv::Rect2f> &boxes,
        const std::vector<int> &ids
) {
    for (auto &entry : trajectories_) {
        entry.second.missed_frames++;
    }

    for (size_t i = 0; i < boxes.size(); ++i) {
        auto &history = trajectories_[ids[i]];
        history.missed_frames = 0;
        history.points.emplace_back(
                boxes[i].x + boxes[i].width * 0.5f,
                boxes[i].y + boxes[i].height * 0.5f
        );
        if (history.points.size() > kTrajectoryHistorySize) {
            history.points.pop_front();
        }
    }

    for (auto it = trajectories_.begin(); it != trajectories_.end();) {
        if (it->second.missed_frames > cfg_.max_missed_frames) {
            it = trajectories_.erase(it);
        } else {
            ++it;
        }
    }
}

void StaticBoxManager::on_mouse_click(
        int x, int y,
        const std::vector <cv::Rect2f> &boxes,
        const std::vector<int> &ids
) {
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (boxes[i].contains(cv::Point2f((float) x, (float) y))) {
            if (boxes_.size() >= static_boxes_max_amount) {
                boxes_.clear();
            }
            static_box sb;
            sb.id = next_id_++;
            sb.rect = boxes[i];
            sb.last_dynamic_id = ids[i];
            sb.confidence = 1.0f;
            sb.state = static_box_state::attached;
            sb.last_seen = std::chrono::steady_clock::now();
            boxes_.push_back(sb);
            return;
        }
    }
}

void StaticBoxManager::update(
        const std::vector <cv::Rect2f> &boxes,
        const std::vector<int> &ids
) {
    auto now = std::chrono::steady_clock::now();

    update_trajectories(boxes, ids);

    for (auto &sb: boxes_) {
        int parent_index = -1;
        for (size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] == sb.last_dynamic_id) {
                parent_index = static_cast<int>(i);
                break;
            }
        }

        if (parent_index >= 0) {
            sb.rect = boxes[static_cast<size_t>(parent_index)];
            sb.last_seen = now;
            sb.missed_frames = 0;
            sb.state = static_box_state::attached;
            sb.confidence = std::min(1.0f, sb.confidence + 0.1f);
            continue;
        }

        sb.missed_frames++;

        if (sb.missed_frames <= cfg_.max_missed_frames) {
            sb.state = static_box_state::lost;
            sb.confidence = std::max(0.0f, sb.confidence - 0.05f);
            continue;
        }

        if (cfg_.auto_rebind) {
            cv::Point2f reference_dir(0.0f, 0.0f);
            bool has_reference_dir = false;
            auto parent_history = trajectories_.find(sb.last_dynamic_id);
            if (parent_history != trajectories_.end() &&
                parent_history->second.points.size() >= 2) {
                const auto &points = parent_history->second.points;
                reference_dir = points.back() - points.front();
                has_reference_dir = (std::abs(reference_dir.x) > 1e-3f ||
                                     std::abs(reference_dir.y) > 1e-3f);
            }

            int best = find_nearest_with_direction(sb, boxes, ids, reference_dir, has_reference_dir);
            if (best >= 0) {
                sb.rect = boxes[best];
                sb.last_dynamic_id = ids[best];
                sb.last_seen = now;
                sb.missed_frames = 0;
                sb.state = static_box_state::pending_rebind;
                sb.confidence = 0.5f;
                continue;
            }
        }

        sb.state = static_box_state::lost;
        sb.confidence = std::max(0.0f, sb.confidence - 0.05f);
    }
}

int StaticBoxManager::find_nearest_with_direction(
        const static_box &sb,
        const std::vector <cv::Rect2f> &boxes,
        const std::vector<int> &ids,
        const cv::Point2f &reference_dir,
        bool has_reference_dir
) const {
    constexpr float kNearbyDistanceEpsilon = 10.0f;
    int best = -1;
    int best_dir = -1;
    float best_dist = std::numeric_limits<float>::max();
    float best_dir_score = -1.0f;
    float best_dir_score_qualified = -1.0f;

    for (size_t i = 0; i < boxes.size(); ++i) {
        float d = center_dist(sb.rect, boxes[i]);
        float dir_score = -1.0f;

        if (has_reference_dir) {
            auto history_it = trajectories_.find(ids[i]);
            if (history_it != trajectories_.end() &&
                history_it->second.points.size() >= 2) {
                const auto &points = history_it->second.points;
                cv::Point2f candidate_dir = points.back() - points.front();
                float ref_norm = std::sqrt(reference_dir.x * reference_dir.x +
                                           reference_dir.y * reference_dir.y);
                float cand_norm = std::sqrt(candidate_dir.x * candidate_dir.x +
                                            candidate_dir.y * candidate_dir.y);
                if (ref_norm > 1e-3f && cand_norm > 1e-3f) {
                    float dot = reference_dir.x * candidate_dir.x + reference_dir.y * candidate_dir.y;
                    dir_score = dot / (ref_norm * cand_norm);
                }
            }
        }

        if (d + kNearbyDistanceEpsilon < best_dist) {
            best_dist = d;
            best = static_cast<int>(i);
            best_dir_score = dir_score;
            best_dir = -1;
            best_dir_score_qualified = -1.0f;
            if (dir_score >= kDirectionSimilarityThreshold) {
                best_dir = static_cast<int>(i);
                best_dir_score_qualified = dir_score;
            }
            continue;
        }

        if (std::abs(d - best_dist) <= kNearbyDistanceEpsilon) {
            if (dir_score > best_dir_score) {
                best_dir_score = dir_score;
            }
            if (dir_score >= kDirectionSimilarityThreshold) {
                if (best_dir == -1 || dir_score > best_dir_score_qualified) {
                    best_dir = static_cast<int>(i);
                    best_dir_score_qualified = dir_score;
                }
            }
        }
    }
    return (best_dir >= 0) ? best_dir : best;
}


bool StaticBoxManager::load_static_rebind_config(const toml::table &tbl) {
// ------------------------- [static_rebind] ------------------------
    try {
        const auto *static_rebind = tbl["static_rebind"].as_table();
        if (!static_rebind) {
            throw std::runtime_error("missing [static_rebind] table");
        }
        cfg_.auto_rebind = read_required<bool>(*static_rebind, "auto_rebind"); //TODO
        cfg_.rebind_timeout_ms = read_required<int>(*static_rebind, "rebind_timeout_ms");
        cfg_.distance_weight = read_required<float>(*static_rebind, "distance_weight"); //TODO
        cfg_.area_weight = read_required<float>(*static_rebind, "area_weight"); //TODO
        cfg_.larger_area_factor = read_required<float>(*static_rebind, "larger_area_factor"); //TODO
        cfg_.max_large_target_dist_frac = read_required<float>(*static_rebind, "max_large_target_dist_frac"); //TODO
        cfg_.parent_iou_th = read_required<float>(*static_rebind, "parent_iou_th");
        cfg_.reattach_score_th = read_required<float>(*static_rebind, "reattach_score_th");

        const auto *tracker = tbl["tracker"].as_table();
        if (!tracker) {
            throw std::runtime_error("missing [tracker] table");
        }
        cfg_.max_missed_frames = read_required<int>(*tracker, "max_missed_frames");
        return true;

    } catch (const std::exception &e) {
        std::cerr << "static_rebind config load failed  " << e.what() << std::endl;
        return false;
    }
};
