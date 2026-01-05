#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <unordered_map>
#include "config.h"

enum class static_box_state {
    attached,          // уверенно привязан
    pending_rebind,    // кандидат найден, ожидаем подтверждения (future)
    lost               // цель потеряна
};

struct static_box {
    int id; // - id: идентификатор статического бокса.
    cv::Rect2f rect; // - rect: текущий bbox статической цели.

    int last_dynamic_id; // - last_dynamic_id: id динамической цели, к которой привязан бокс.
    float confidence; // - confidence: уверенность привязки.
    int missed_frames = 0; // - missed_frames: число кадров без обновления.

    static_box_state state; // - state: текущее состояние статической цели.
};


class StaticBoxManager {
public:
    // Создаёт менеджер статических боксов и загружает конфигурацию.
    explicit StaticBoxManager(const toml::table &tbl);

    // Обрабатывает клик мыши и создаёт статический бокс.
    void on_mouse_click(
            int x, int y,
            const std::vector <cv::Rect2f> &dynamic_boxes,
            const std::vector<int> &dynamic_ids
    );

    // Обновляет статические боксы с учётом динамических целей.
    void update(
            const std::vector <cv::Rect2f> &dynamic_boxes,
            const std::vector<int> &dynamic_ids
    );

    // Возвращает список текущих статических боксов.
    const std::vector <static_box> &boxes() const { return boxes_; }

    // Заглушка для будущего расширения функциональности статического менеджера.
    void static_mgr(); //TODO реализовать!


private:
    struct StaticBoxConfig {
// ============================================================================
// Static rebind configuration
// ============================================================================
        // Автоматическая перепривязка static bbox
        bool AUTO_REBIND = true; // - AUTO_REBIND: включение автоматической перепривязки.

        // Максимально допустимое количество пропущенных кадров
        int MAX_MISSED_FRAMES = 3; // - MAX_MISSED_FRAMES: допустимое количество пропусков кадров.

        // Максимальное количество статических боксов
        int MAX_STATIC_BOXES = 1; // - MAX_STATIC_BOXES: максимум одновременно активных статических боксов.

        // Длина истории траекторий
        int TRAJECTORY_HISTORY_SIZE = 8; // - TRAJECTORY_HISTORY_SIZE: длина истории траекторий (кадры).

        // Порог похожести направления движения
        float DIRECTION_SIMILARITY_THRESHOLD = 0.5f; // - DIRECTION_SIMILARITY_THRESHOLD: порог похожести направления.

        // Эпсилон для сравнения расстояний
        float NEARBY_DISTANCE_EPSILON = 10.0f; // - NEARBY_DISTANCE_EPSILON: допуск по расстоянию (пиксели).
    };

    struct TrajectoryHistory {
        std::deque<cv::Point2f> points; // - points: история центров динамической цели.
        int missed_frames = 0; // - missed_frames: количество кадров без обновления истории.
    };

    StaticBoxConfig cfg_; // - cfg_: конфигурация статических боксов.
    std::vector <static_box> boxes_; // - boxes_: текущие статические боксы.
    int next_id_ = 1; // - next_id_: счётчик идентификаторов.
    std::unordered_map<int, TrajectoryHistory> trajectories_; // - trajectories_: истории движения динамических целей.

    // Обновляет историю траекторий на основе динамических целей.
    void update_trajectories(
            const std::vector<cv::Rect2f> &boxes,
            const std::vector<int> &ids
    );

    // Ищет ближайший динамический бокс с учётом направления движения.
    int find_nearest_with_direction(
            const static_box &sb,
            const std::vector <cv::Rect2f> &boxes,
            const std::vector<int> &ids,
            const cv::Point2f &reference_dir,
            bool has_reference_dir
    ) const;

    // Загружает параметры перепривязки статических боксов из TOML.
    bool load_static_rebind_config(const toml::table &tbl);
};
