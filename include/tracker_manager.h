#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "target.h"
#include "config.h"

class TrackerManager {
private:
    struct TrackerConfig {
        // IoU для сопоставления
        float iou_threshold = 0.25f;
        // Максимальное число пропущенных кадров без детекции
        int max_missed_frames = 30;
        // Максимумальное кол-во отслеживаемых активных целей
        int max_targets = 10;
        // Оставлять только "ведущую" цель по направлению движения
        bool leading_only = false;
        // Минимальная скорость для учёта направления (пикселей за кадр)
        float leading_min_speed = 2.0f;
    };


    struct Track {
        int id = -1;
        cv::Rect2f bbox;
        int age = 0;
        int missed = 0;
        cv::Point2f last_center{0.0f, 0.0f};
        cv::Point2f velocity{0.0f, 0.0f};
        bool has_center = false;
    };

    static float iou(const cv::Rect2f &a, const cv::Rect2f &b);
    TrackerConfig cfg_;
    int next_id_ = 1;
    std::vector<Track> tracks_;
    std::vector<Target> targets_;
    int leading_id_ = -1;

    bool load_tracker_config(const toml::table& tbl);

public:
    explicit TrackerManager(const toml::table &tbl);

    void reset();

    std::vector<Target> update(const std::vector<cv::Rect2f> &detections);

    int pickTargetId(int x, int y) const;

    bool hasTargetId(int id) const;

    const std::vector<Target> &targets() const { return targets_; }


};
