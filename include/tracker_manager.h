#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "target.h"
#include "config.h"

class TrackerManager {
private:
    struct TrackerConfig {
        // IoU для сопоставления
        float IOU_THRESHOLD = 0.25f; // - IOU_THRESHOLD: минимальный IoU для ассоциации детекций.
        // Максимальное число пропущенных кадров без детекции
        int MAX_MISSED_FRAMES = 30; // - MAX_MISSED_FRAMES: допустимое число пропусков кадров.
        // Максимумальное кол-во отслеживаемых активных целей
        int MAX_TARGETS = 10; // - MAX_TARGETS: максимальное число активных целей.
        // Оставлять только "ведущую" цель по направлению движения
        bool LEADING_ONLY = false; // - LEADING_ONLY: оставлять только ведущую цель.
        // Минимальная скорость для учёта направления (пикселей за кадр)
        float LEADING_MIN_SPEED = 2.0f; // - LEADING_MIN_SPEED: порог скорости для расчёта направления.
    };


    struct Track {
        int id = -1; // - id: идентификатор трека.
        cv::Rect2f bbox; // - bbox: текущий bbox трека.
        int missed = 0; // - missed: количество пропущенных кадров.
        cv::Point2f last_center{0.0f, 0.0f}; // - last_center: последняя позиция центра.
        cv::Point2f velocity{0.0f, 0.0f}; // - velocity: оценка скорости.
        bool has_center = false; // - has_center: был ли рассчитан центр.
    };

    // Рассчитывает IoU для двух прямоугольников.
    static float iou(const cv::Rect2f &a, const cv::Rect2f &b);
    TrackerConfig cfg_; // - cfg_: конфигурация трекера.
    int next_id_ = 1; // - next_id_: счётчик id для новых треков.
    std::vector<Track> tracks_; // - tracks_: активные треки.
    std::vector<Target> targets_; // - targets_: список целей для выдачи наружу.
    int leading_id_ = -1; // - leading_id_: id ведущей цели.

    // Загружает конфигурацию трекера из TOML.
    bool load_tracker_config(const toml::table& tbl);

public:
    // Создаёт менеджер трекера и загружает конфигурацию.
    explicit TrackerManager(const toml::table &tbl);

    // Сбрасывает состояние трекера и очищает все цели.
    void reset();

    // Обновляет треки на основе списка детекций и возвращает цели.
    std::vector<Target> update(const std::vector<cv::Rect2f> &detections);

    // Возвращает текущий список целей.
    const std::vector<Target> &targets() const { return targets_; }


};
