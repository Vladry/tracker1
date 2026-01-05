#pragma once

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
        int MAX_TARGETS = 5; // - MAX_TARGETS: максимум активных целей.
        int CLICK_PADDING = 6; // - CLICK_PADDING: пиксели вокруг bbox, в которых клик удаляет цель.
        int MOTION_DIFF_THRESHOLD = 25; // - MOTION_DIFF_THRESHOLD: порог бинаризации diff-кадра для движения.
        int CLICK_CAPTURE_SIZE = 80; // - CLICK_CAPTURE_SIZE: размер ROI для анализа движения вокруг клика.
        int MOTION_FRAMES = 3; // - MOTION_FRAMES: число кадров для анализа движения.
        float MOTION_ANGLE_TOLERANCE_DEG = 20.0f; // - MOTION_ANGLE_TOLERANCE_DEG: допуск угла движения (градусы).
        int OVERLAY_TTL_SECONDS = 3; // - OVERLAY_TTL_SECONDS: время жизни красного оверлея после создания цели.
        int TRACKER_INIT_PADDING = 10; // - TRACKER_INIT_PADDING: расширение bbox перед запуском трекера.
        int TRACKER_MIN_SIZE = 24; // - TRACKER_MIN_SIZE: минимальный размер bbox перед запуском трекера.
        float MOTION_MIN_MAGNITUDE = 0.4f; // - MOTION_MIN_MAGNITUDE: минимальная средняя скорость движения.
        float MOTION_MAG_TOLERANCE_PX = 3.0f; // - MOTION_MAG_TOLERANCE_PX: допуск по длине шага движения.
        int MOTION_MAX_FEATURES = 200; // - MOTION_MAX_FEATURES: максимум ключевых точек.
        float MOTION_QUALITY_LEVEL = 0.01f; // - MOTION_QUALITY_LEVEL: порог качества keypoints.
        float MOTION_MIN_DISTANCE = 3.0f; // - MOTION_MIN_DISTANCE: минимальная дистанция keypoints.
        float MOTION_ANGLE_BIN_DEG = 10.0f; // - MOTION_ANGLE_BIN_DEG: размер бина направлений.
        float MOTION_MAG_BIN_PX = 2.0f; // - MOTION_MAG_BIN_PX: размер бина длины шага.
        float MOTION_GRID_STEP_RATIO = 0.1f; // - MOTION_GRID_STEP_RATIO: шаг сетки (доля ROI).
        float MOTION_MIN_STABLE_RATIO = 0.1f; // - MOTION_MIN_STABLE_RATIO: доля стабильных пикселей.
        int WATCHDOG_PERIOD_MS = 100; // - WATCHDOG_PERIOD_MS: период проверки группового движения по ROI (мс).
        float WATCHDOG_MOTION_RATIO = 0.8f; // - WATCHDOG_MOTION_RATIO: доля площади ROI с едино-направленным движением.
        float WATCHDOG_ANGLE_TOLERANCE_DEG = 20.0f; // - WATCHDOG_ANGLE_TOLERANCE_DEG: допуск угла движения (градусы).
        int VISIBILITY_HISTORY_SIZE = 3; // - VISIBILITY_HISTORY_SIZE: размер буфера видимости для фильтра потери.
        int RESERVED_CANDIDATE_TTL_MS = 1500; // - RESERVED_CANDIDATE_TTL_MS: TTL резерва кандидатов (мс).
        int MOTION_DETECTION_ITERATIONS = 10; // - MOTION_DETECTION_ITERATIONS: число итераций детекции движения.
        float MOTION_DETECTION_DIFFUSION_PX = 100.0f; // - MOTION_DETECTION_DIFFUSION_PX: радиус кластеризации детекций.
        float MOTION_DETECTION_CLUSTER_RATIO = 0.9f; // - MOTION_DETECTION_CLUSTER_RATIO: доля детекций в кластере.
        int AUTO_HISTORY_SIZE = 5; // - AUTO_HISTORY_SIZE: история кадров для автодетекции движения.
        int AUTO_DIFF_THRESHOLD = 25; // - AUTO_DIFF_THRESHOLD: порог бинаризации diff для автодетекции.
        double AUTO_MIN_AREA = 60.0; // - AUTO_MIN_AREA: минимальная площадь контура для автодетекции.
        bool FLOODFILL_FILL_OVERLAY = true; // - FLOODFILL_FILL_OVERLAY: включение оверлея для визуализации зоны движения.
        float FLOODFILL_OVERLAY_ALPHA = 0.7f; // - FLOODFILL_OVERLAY_ALPHA: альфа заливки floodfill оверлея.
        int MIN_AREA = 60; // - MIN_AREA: минимальная площадь ROI для создания трека.
        int MIN_WIDTH = 10; // - MIN_WIDTH: минимальная ширина ROI.
        int MIN_HEIGHT = 10; // - MIN_HEIGHT: минимальная высота ROI.
        std::string TRACKER_TYPE = "KCF"; // - TRACKER_TYPE: имя OpenCV-трекера (KCF/CSRT).
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
        std::vector<bool> visibility_history; // - visibility_history: история видимости для фильтра потери.
        size_t visibility_index = 0; // - visibility_index: индекс кольцевого буфера истории видимости.
        cv::Point2f last_known_center{0.0f, 0.0f}; // - last_known_center: последняя известная позиция центра цели.
        cv::Point2f cross_center{0.0f, 0.0f}; // - cross_center: центр красного крестика цели.
        AutoCandidateSearch candidate_search; // - candidate_search: авто-поиск кандидата при потере цели.
    };

    struct PendingClick {
        cv::Rect roi; // - roi: ROI вокруг клика для проверки движения.
        std::vector<cv::Mat> gray_frames; // - gray_frames: последовательность кадров для анализа движения.
    };

    struct ReservedCandidate {
        cv::Rect2f bbox; // - bbox: зарезервированный bbox кандидата.
        long long expires_ms = 0; // - expires_ms: время истечения резерва.
    };

    Config cfg_; // - cfg_: текущие настройки ручного трекера.
    LoggingConfig log_cfg_; // - log_cfg_: настройки логирования.
    std::vector<ManualTrack> tracks_; // - tracks_: список активных треков.
    std::vector<Target> targets_; // - targets_: список целей для выдачи наружу.
    std::vector<PendingClick> pending_clicks_; // - pending_clicks_: клики, ожидающие подтверждения движения.
    std::vector<ReservedCandidate> reserved_candidates_; // - reserved_candidates_: временные резервы кандидатов.
    int next_id_ = 1; // - next_id_: счётчик id целей.
    std::mutex mutex_; // - mutex_: защита данных от конкурентного доступа.
    cv::Mat flood_fill_overlay_; // - flood_fill_overlay_: визуальный оверлей зоны движения.
    cv::Mat flood_fill_mask_; // - flood_fill_mask_: маска оверлея зоны движения.
    long long overlay_expire_ms_ = 0; // - overlay_expire_ms_: время истечения оверлея.
    cv::Mat watchdog_prev_gray_; // - watchdog_prev_gray_: кадр для контроля группового движения.
    long long watchdog_prev_ms_ = 0; // - watchdog_prev_ms_: время предыдущей проверки движения.
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
    // Проверяет наличие синхронного группового движения в ROI трека.
    bool has_group_motion(const cv::Mat& prev_gray, const cv::Mat& curr_gray, const cv::Rect2f& roi) const;
    // Переводит трек в состояние потери (серый bbox + поиск кандидата).
    void mark_track_lost(ManualTrack& track, long long now_ms);
    // Обновляет публичный список целей на основе внутренних треков.
    void refresh_targets();
};
