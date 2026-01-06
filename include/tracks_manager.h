#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "target.h"
#include "config.h"

class TracksManager {
private:
    struct TrackerConfig {
        float IOU_THRESHOLD = 0.25f; // - минимальный IoU для выполнения детекций.
        int MAX_MISSED_FRAMES = 30; // Допустимое максимальное число пропущенных кадров без детекции.
        int MAX_TARGETS = 10; // Максимумальное кол-во отслеживаемых активных целей.
        bool LEADING_ONLY = false; // Оставлять только "ведущую" цель по направлению движения.
        float LEADING_MIN_SPEED = 2.0f; // - порог скорости для расчёта направления. Минимальная скорость для учёта направления (пикселей за кадр)
    };


    struct Track {
        int id = -1; // - идентификатор трека.
        cv::Rect2f bbox; // - текущий bbox трека.
        int missed = 0; // - количество пропущенных кадров.
        cv::Point2f last_center{0.0f, 0.0f}; // - последняя позиция центра.
        cv::Point2f velocity{0.0f, 0.0f}; // - оценка скорости.
        bool has_center = false; // - был ли рассчитан центр.
    };

    // Рассчитывает IoU для двух прямоугольников.
    static float iou(const cv::Rect2f &a, const cv::Rect2f &b);
    TrackerConfig cfg_; // - конфигурация трекера.
    int next_id_ = 1; // - счётчик id для новых треков.
    std::vector<Track> tracks_; // - активные треки.
    std::vector<Target> targets_; // - список целей для выдачи наружу.
    int leading_id_ = -1; // - id вдущей цели.

    // Загружает конфигурацию трекера из TOML.
    bool load_tracker_config(const toml::table& tbl);

public:
    // Создаёт менеджер трекера и загружает конфигурацию.
    explicit TracksManager(const toml::table &tbl);

    // Сбрасывает состояние трекера и очищает все цели.
    void reset();

    // Обновляет треки на основе списка детекций и возвращает цели.
    std::vector<Target> update(const std::vector<cv::Rect2f> &detections);

    // Возвращает текущий список целей.
    const std::vector<Target> &targets() const { return targets_; }


};
