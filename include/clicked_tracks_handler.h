#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <mutex>
#include <string>
#include <vector>
#include "auto_candidate_search.h"
#include "motion_detector.h"
#include "config.h"
#include "clicked_target_shaper.h"
#include "target.h"

class ClickedTracksHandler {
public:
    struct Config {
        int MAX_TARGETS = 5; // - максимум активных целей.
        int CLICK_PADDING = 6; // - пиксели вокруг bbox, в которых клик удаляет цель.
        int MOTION_DIFF_THRESHOLD = 25; // - порог бинаризации diff-кадра для движения.
        int CLICK_CAPTURE_SIZE = 80; // - размер ROI для анализа движения вокруг клика.
        int MOTION_FRAMES = 3; // - число кадров для анализа движения.
        float MOTION_ANGLE_TOLERANCE_DEG = 20.0f; // - допуск угла движения (градусы).
        int OVERLAY_TTL_SECONDS = 3; // - время жизни красного оверлея после создания цели.
        int TRACKER_INIT_PADDING = 10; // - расширение bbox перед запуском трекера.
        int TRACKER_MIN_SIZE = 24; // - минимальный размер bbox перед запуском трекера.
        float MOTION_MIN_MAGNITUDE = 0.4f; // - минимальная средняя скорость движения.
        float MOTION_MAG_TOLERANCE_PX = 3.0f; // - допуск по длине шага движения.
        int MOTION_MAX_FEATURES = 200; // - максимум ключевых точек.
        float MOTION_QUALITY_LEVEL = 0.01f; // - порог качества keypoints.
        float MOTION_MIN_DISTANCE = 3.0f; // - минимальная дистанция keypoints.
        float MOTION_ANGLE_BIN_DEG = 10.0f; // - размер бина направлений.
        float MOTION_MAG_BIN_PX = 2.0f; // - размер бина длины шага.
        float MOTION_GRID_STEP_RATIO = 0.1f; // - шаг сетки (доля ROI).
        float MOTION_MIN_STABLE_RATIO = 0.1f; // - доля стабильных пикселей.
        int WATCHDOG_PERIOD_MS = 100; // - период проверки группового движения по ROI (мс).
        float WATCHDOG_MOTION_RATIO = 0.8f; // - доля площади ROI с едино-направленным движением.
        float WATCHDOG_ANGLE_TOLERANCE_DEG = 20.0f; // - допуск угла движения (градусы).
        double WATCHDOG_FLOW_PYR_SCALE = 0.5; // - параметр pyr_scale для calcOpticalFlowFarneback.
        int WATCHDOG_FLOW_LEVELS = 3; // - число уровней пирамиды в calcOpticalFlowFarneback.
        int WATCHDOG_FLOW_WINSIZE = 15; // - размер окна для calcOpticalFlowFarneback.
        int WATCHDOG_FLOW_ITERATIONS = 3; // - число итераций на уровне в calcOpticalFlowFarneback.
        int WATCHDOG_FLOW_POLY_N = 5; // - размер окрестности для аппроксимации в calcOpticalFlowFarneback.
        double WATCHDOG_FLOW_POLY_SIGMA = 1.2; // - sigma для аппроксимации полинома в calcOpticalFlowFarneback.
        int WATCHDOG_FLOW_FLAGS = 0; // - флаги для calcOpticalFlowFarneback.
        int VISIBILITY_HISTORY_SIZE = 3; // - размер буфера видимости для фильтра потери.
        int RESERVED_CANDIDATE_TTL_MS = 1500; // - TTL резерва кандидатов (мс).
        float RESERVED_DETECTION_RADIUS_PX = 200.0f; // - радиус запрета вокруг зарезервированных кандидатов.
        int AUTO_DETECTION_PERIOD_MS = 100; // - период обновления автодетекции (мс).
        int MOTION_DETECTION_ITERATIONS = 10; // - число итераций детекции движения.
        float MOTION_DETECTION_DIFFUSION_PX = 100.0f; // - радиус кластеризации детекций.
        float MOTION_DETECTION_CLUSTER_RATIO = 0.9f; // - доля детекций в кластере.
        int AUTO_HISTORY_SIZE = 5; // - история кадров для автодетекции движения.
        int AUTO_DIFF_THRESHOLD = 25; // - порог бинаризации diff для автодетекции.
        int MOTION_BINARY_MAX_VALUE = 255; // - максимальное значение пикселя при бинаризации diff-кадра.
        double AUTO_MIN_AREA = 60.0; // - минимальная площадь контура для автодетекции.
        bool FLOODFILL_FILL_OVERLAY = true; // - включение оверлея для визуализации зоны движения.
        float FLOODFILL_OVERLAY_ALPHA = 0.7f; // - альфа заливки floodfill оверлея.
        int MIN_AREA = 60; // - минимальная площадь ROI для создания трека.
        int MIN_WIDTH = 10; // - минимальная ширина ROI.
        int MIN_HEIGHT = 10; // - минимальная высота ROI.
        std::string TRACKER_TYPE = "KCF"; // - имя OpenCV-трекера (KCF/CSRT).
    };

    // Создаёт менеджер ручного трекинга и загружает конфигурацию из TOML.
    explicit ClickedTracksHandler(const toml::table& tbl);

    // Обновляет активные кликнутые мышью треки и синхронизирует список целей.
    void update(cv::Mat& frame, long long now_ms);
    // Обрабатывает клик мыши и создаёт/удаляет кликнутую мышью цель.
    bool handle_click(int x, int y, const cv::Mat& frame, long long now_ms);
    // Возвращает текущее представление целей для рендера/выдачи.
    const std::vector<Target>& targets() const { return targets_; }

private:
    struct ClickedTrack {
        int id = -1; // - идентификатор цели.
        cv::Rect2f bbox; // - текущий bbox цели.
        cv::Ptr<cv::Tracker> tracker; // - экземпляр OpenCV-трекера.
        long long lost_since_ms = 0; // - время начала потери цели (0 — цель видна).
        std::vector<bool> visibility_history; // - история видимости для фильтра потери.
        size_t visibility_index = 0; // - индекс кольцевого буфера истории видимости.
        cv::Point2f last_known_center{0.0f, 0.0f}; // - последняя известная позиция центра цели.
        cv::Point2f cross_center{0.0f, 0.0f}; // - центр красного крестика цели.
        AutoCandidateSearch candidate_search; // - авто-поиск кандидата при потере цели.
    };

    struct PendingClick {
        cv::Rect roi; // - ROI вокруг клика для проверки движения.
        std::vector<cv::Mat> gray_frames; // - последовательность кадров для анализа движения.
    };

    struct ReservedCandidate {
        cv::Rect2f bbox; // - зарезервированный bbox кандидата.
        long long expires_ms = 0; // - время истечения резерва.
        int owner_id = -1; // - id трека, которому назначен кандидат.
    };

    Config cfg_; // - текущие настройки ручного трекера.
    LoggingConfig log_cfg_; // - настройки логирования.
    std::vector<ClickedTrack> tracks_; // - список активных треков.
    std::vector<Target> targets_; // - список целей для выдачи наружу.
    std::vector<PendingClick> pending_clicks_; // - клики, ожидающие подтверждения движения.
    std::vector<ReservedCandidate> reserved_candidates_; // - детекции_кандидаты, временно закреплённые за треками которым их выдали.
    int next_id_ = 1; // - счётчик id целей.
    std::mutex mutex_; // - защита данных от конкурентного доступа.
    cv::Mat flood_fill_overlay_; // - визуальный оверлей зоны движения.
    cv::Mat flood_fill_mask_; // - маска оверлея зоны движения.
    long long overlay_expire_ms_ = 0; // - время истечения оверлея.
    cv::Mat watchdog_prev_gray_; // - кадр для контроля группового движения.
    long long watchdog_prev_ms_ = 0; // - время предыдущей проверки движения.
    ClickedTargetShaper clicked_target_shaper_; // - формирователь цели по клику.
    MotionDetector motion_detector_; // - фоновый детектор движения (пул кандидатов).

    // Загружает параметры ручного трекера из TOML.
    bool load_config(const toml::table& tbl);
    // Создаёт OpenCV-трекер согласно настройкам.
    cv::Ptr<cv::Tracker> create_tracker() const;
    // Проверяет попадание точки в bbox с дополнительным паддингом.
    bool point_in_rect_with_padding(const cv::Rect2f& rect, int x, int y, int pad) const;
    // Записывает состояние видимости в кольцевой буфер.
    void record_visibility(ClickedTrack& track, bool visible);
    // Проверяет, что цель была невидима в последние несколько кадров.
    bool has_recent_visibility_loss(const ClickedTrack& track) const;
    // Проверяет наличие синхронного группового движения в ROI трека.
    bool has_group_motion(const cv::Mat& prev_gray, const cv::Mat& curr_gray, const cv::Rect2f& roi) const;
    // Переводит трек в состояние потери (серый bbox + поиск кандидата).
    void mark_track_lost(ClickedTrack& track, long long now_ms);
    // Обновляет публичный список целей на основе внутренних треков.
    void refresh_targets();
};
