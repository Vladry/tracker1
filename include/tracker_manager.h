#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "target.h"
#include "config.h"

class TrackerManager {
private:
    struct TrackerConfig {
        // IoU для сопоставления
        float iou_threshold = 0.25f; // - iou_threshold: минимальный IoU для ассоциации детекций.
        // Максимальное число пропущенных кадров без детекции
        int max_missed_frames = 30; // - max_missed_frames: допустимое число пропусков кадров.
        // Максимумальное кол-во отслеживаемых активных целей
        int max_targets = 10; // - max_targets: максимальное число активных целей.
        // Оставлять только "ведущую" цель по направлению движения
        bool leading_only = false; // - leading_only: оставлять только ведущую цель.
        // Минимальная скорость для учёта направления (пикселей за кадр)
        float leading_min_speed = 2.0f; // - leading_min_speed: порог скорости для расчёта направления.
        // Использовать Калман для предсказания позиции
        bool use_kalman = true; // - use_kalman: включение фильтра Калмана.
        // Шум процесса для Калмана (больше -> быстрее реагирует)
        float kalman_process_noise = 1e-2f; // - kalman_process_noise: шум процесса Калмана.
        // Шум измерений для Калмана (больше -> сильнее сглаживание)
        float kalman_measurement_noise = 1e-1f; // - kalman_measurement_noise: шум измерений Калмана.
    };


    struct Track {
        int id = -1; // - id: идентификатор трека.
        cv::Rect2f bbox; // - bbox: текущий bbox трека.
        int age = 0; // - age: количество кадров жизни.
        int missed = 0; // - missed: количество пропущенных кадров.
        cv::Point2f last_center{0.0f, 0.0f}; // - last_center: последняя позиция центра.
        cv::Point2f velocity{0.0f, 0.0f}; // - velocity: оценка скорости.
        bool has_center = false; // - has_center: был ли рассчитан центр.
        cv::KalmanFilter kf; // - kf: фильтр Калмана для предсказания.
        bool kf_ready = false; // - kf_ready: инициализирован ли фильтр Калмана.
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
    // Инициализирует фильтр Калмана для трека.
    void init_kalman(Track &track, const cv::Point2f &center) const;
    // Прогнозирует позицию трека фильтром Калмана.
    void predict_kalman(Track &track);
    // Корректирует фильтр Калмана измерением центра.
    void correct_kalman(Track &track, const cv::Point2f &center);

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
