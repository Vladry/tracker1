#include "static_box_manager.h"
#include <cmath>
#include <limits>

static float iou_rect(const cv::Rect2f &a, const cv::Rect2f &b) {
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    return (uni > 0.f) ? (inter / uni) : 0.f;
}

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

void StaticBoxManager::on_mouse_click(
        int x, int y,
        const std::vector <cv::Rect2f> &boxes,
        const std::vector<int> &ids
) {
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (boxes[i].contains(cv::Point2f((float) x, (float) y))) {
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

    for (auto &sb: boxes_) {
        float score = 0.f;
        int best = find_best_match(sb, boxes, ids, score);

        if (best >= 0 && score >= cfg_.reattach_score_th) {
            sb.rect = boxes[best];
            sb.last_dynamic_id = ids[best];
            sb.last_seen = now;
            sb.state = static_box_state::attached;
            sb.confidence = std::min(1.0f, sb.confidence + 0.1f);
            continue;
        }

        auto lost_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - sb.last_seen).count();

        if (cfg_.auto_rebind_on_loss && lost_ms > cfg_.rebind_timeout_ms) {
            int nearest = find_nearest(sb, boxes);
            if (nearest >= 0) {
                sb.rect = boxes[nearest];
                sb.last_dynamic_id = ids[nearest];
                sb.last_seen = now;
                sb.state = static_box_state::pending_rebind;
                sb.confidence = 0.5f;
                continue;
            }
        }

        sb.state = static_box_state::lost;
        sb.confidence = std::max(0.0f, sb.confidence - 0.05f);
    }
}

int StaticBoxManager::find_best_match(
        const static_box &sb,
        const std::vector <cv::Rect2f> &boxes,
        const std::vector<int> &,
        float &out_score
) const {
    int best = -1;
    out_score = 0.f;

    for (size_t i = 0; i < boxes.size(); ++i) {
        float v_iou = iou_rect(sb.rect, boxes[i]);
        float d = center_dist(sb.rect, boxes[i]);
        float score = v_iou + 0.001f / (1.0f + d);

        if (score > out_score) {
            out_score = score;
            best = (int) i;
        }
    }
    return best;
}

int StaticBoxManager::find_nearest(
        const static_box &sb,
        const std::vector <cv::Rect2f> &boxes
) const {
    float best_d = std::numeric_limits<float>::max();
    int best = -1;

    for (size_t i = 0; i < boxes.size(); ++i) {
        float d = center_dist(sb.rect, boxes[i]);
        if (d < best_d) {
            best_d = d;
            best = (int) i;
        }
    }
    return best;
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
        return true;

    } catch (const std::exception &e) {
        std::cerr << "static_rebind config load failed  " << e.what() << std::endl;
        return false;
    }
};

void StaticBoxManager::static_mgr(){};// TODO - реализовать!
    /*    работает с константами:
    bool auto_rebind_on_loss,
    int rebind_timeout_ms,
    float parent_iou_th,
    float reattach_score_th*/