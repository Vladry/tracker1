#pragma once

#include <array>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <mutex>
#include <string>
#include <vector>
#include "auto_candidate_search.h"
#include "config.h"
#include "manual_motion_detector.h"
#include "target.h"

class ManualTrackerManager {
public:
    struct Config {
        int max_targets = 5; // - max_targets: максимум активных целей.
        int click_padding = 6; // - click_padding: пиксели вокруг bbox, в которых клик удаляет цель.
        int motion_diff_threshold = 25; // - motion_diff_threshold: порог бинаризации diff-кадра для движения.
        int click_capture_size = 80; // - click_capture_size: размер ROI для анализа движения вокруг клика.
        int motion_frames = 3; // - motion_frames: число кадров для анализа движения.
        int overlay_ttl_seconds = 3; // - overlay_ttl_seconds: время жизни красного оверлея после создания цели.
        int tracker_init_padding = 10; // - tracker_init_padding: расширение bbox перед запуском трекера.
        int tracker_min_size = 24; // - tracker_min_size: минимальный размер bbox перед запуском трекера.
        float motion_min_magnitude = 0.4f; // - motion_min_magnitude: минимальная средняя скорость движения.
        float motion_mag_tolerance_px = 3.0f; // - motion_mag_tolerance_px: допуск по длине шага движения.
        bool floodfill_fill_overlay = true; // - floodfill_fill_overlay: включение оверлея для визуализации зоны движения.
        int floodfill_lo_diff = 20; // - floodfill_lo_diff: нижний порог flood fill (зарезервировано).
        int floodfill_hi_diff = 20; // - floodfill_hi_diff: верхний порог flood fill (зарезервировано).
        int min_area = 200; // - min_area: минимальная площадь ROI для создания трека.
        int min_width = 10; // - min_width: минимальная ширина ROI.
        int min_height = 10; // - min_height: минимальная высота ROI.
        std::string tracker_type = "KCF"; // - tracker_type: имя OpenCV-трекера (KCF/CSRT).
    };

    // Создаёт менеджер ручного трекинга и загружает конфигурацию из TOML.
    explicit ManualTrackerManager(const toml::table& tbl);

    // Обновляет активные ручные треки и синхронизирует список целей.
    void update(cv::Mat& frame, long long now_ms);
    // Обрабатывает клик мыши и создаёт/удаляет ручную цель.
    bool handle_click(int x, int y, const cv::Mat& frame, long long now_ms);
    // Возвращает текущее представление целей для рендера/выдачи.
    const std::vector<Target>& targets() const { return targets_; }

private:
    struct ManualTrack {
        int id = -1; // - id: идентификатор цели.
        cv::Rect2f bbox; // - bbox: текущий bbox цели.
        cv::Ptr<cv::Tracker> tracker; // - tracker: экземпляр OpenCV-трекера.
        long long lost_since_ms = 0; // - lost_since_ms: время начала потери цели (0 — цель видна).
        std::array<bool, 3> visibility_history{true, true, true}; // - visibility_history: история видимости для фильтра потери.
        size_t visibility_index = 0; // - visibility_index: индекс кольцевого буфера истории видимости.
        cv::Point2f last_known_center{0.0f, 0.0f}; // - last_known_center: последняя известная позиция центра цели.
        AutoCandidateSearch candidate_search; // - candidate_search: авто-поиск кандидата при потере цели.
    };

    struct PendingClick {
        cv::Rect roi; // - roi: ROI вокруг клика для проверки движения.
        std::vector<cv::Mat> gray_frames; // - gray_frames: последовательность кадров для анализа движения.
    };

    Config cfg_; // - cfg_: текущие настройки ручного трекера.
    LoggingConfig log_cfg_; // - log_cfg_: настройки логирования.
    std::vector<ManualTrack> tracks_; // - tracks_: список активных треков.
    std::vector<Target> targets_; // - targets_: список целей для выдачи наружу.
    std::vector<PendingClick> pending_clicks_; // - pending_clicks_: клики, ожидающие подтверждения движения.
    int next_id_ = 1; // - next_id_: счётчик id целей.
    std::mutex mutex_; // - mutex_: защита данных от конкурентного доступа.
    cv::Mat flood_fill_overlay_; // - flood_fill_overlay_: визуальный оверлей зоны движения.
    cv::Mat flood_fill_mask_; // - flood_fill_mask_: маска оверлея зоны движения.
    long long overlay_expire_ms_ = 0; // - overlay_expire_ms_: время истечения оверлея.
    ManualMotionDetector motion_detector_; // - motion_detector_: детектор движения в ROI клика.

    // Загружает параметры ручного трекера из TOML.
    bool load_config(const toml::table& tbl);
    // Создаёт OpenCV-трекер согласно настройкам.
    cv::Ptr<cv::Tracker> create_tracker() const;
    // Проверяет попадание точки в bbox с дополнительным паддингом.
    bool point_in_rect_with_padding(const cv::Rect2f& rect, int x, int y, int pad) const;
    // Записывает состояние видимости в кольцевой буфер.
    void record_visibility(ManualTrack& track, bool visible);
    // Проверяет, что цель была невидима в последние несколько кадров.
    bool has_recent_visibility_loss(const ManualTrack& track) const;
    // Обновляет публичный список целей на основе внутренних треков.
    void refresh_targets();
};
