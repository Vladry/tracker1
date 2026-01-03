#pragma once

#include <opencv2/opencv.hpp>
#include <mutex>
#include <string>
#include <vector>
#include "config.h"

struct StaticTarget {
    int id = -1; // - id: идентификатор статической цели.
    cv::Rect2f bbox; // - bbox: прямоугольник статической цели.
};

class StaticTargetManager {
public:
    struct Config {
        int max_targets = 5; // - max_targets: максимум статических целей.
        int click_padding = 6; // - click_padding: паддинг вокруг ROI клика.
        int remove_padding = 6; // - remove_padding: паддинг удаления цели по клику.
        int fallback_box_size = 40; // - fallback_box_size: размер запасного bbox при ошибке floodfill.
        float max_area_ratio = 0.1f; // - max_area_ratio: максимум площади ROI от площади кадра.
        bool click_equalize = true; // - click_equalize: включение эквализации гистограммы перед floodfill.
        int floodfill_lo_diff = 20; // - floodfill_lo_diff: нижний порог flood fill.
        int floodfill_hi_diff = 20; // - floodfill_hi_diff: верхний порог flood fill.
        int overlay_ttl_seconds = 3; // - overlay_ttl_seconds: время жизни оверлея floodfill.
        int min_area = 60; // - min_area: минимальная площадь ROI.
        int min_width = 6; // - min_width: минимальная ширина ROI.
        int min_height = 6; // - min_height: минимальная высота ROI.
        float min_contrast = 5.0f; // - min_contrast: минимальный контраст ROI.
    };

    // Создаёт менеджер статических целей и загружает конфигурацию.
    explicit StaticTargetManager(const toml::table& tbl);

    // Обрабатывает ПКМ: удаление цели или добавление новой.
    bool handle_right_click(int x, int y, const cv::Mat& frame, long long now_ms);
    // Обновляет оверлей floodfill и устаревшие элементы.
    void update(cv::Mat& frame, long long now_ms);
    // Возвращает список статических целей.
    const std::vector<StaticTarget>& targets() const { return targets_; }

private:
    Config cfg_; // - cfg_: настройки статического детектора.
    LoggingConfig log_cfg_; // - log_cfg_: настройки логирования.
    std::vector<StaticTarget> targets_; // - targets_: список статических целей.
    int next_id_ = 1; // - next_id_: счётчик идентификаторов целей.
    std::mutex mutex_; // - mutex_: защита доступа из разных потоков.

    // Загружает конфигурацию статического детектора из TOML.
    bool load_config(const toml::table& tbl);
    // Строит ROI вокруг клика с использованием floodfill.
    cv::Rect2f build_roi_from_click(const cv::Mat& frame, int x, int y, cv::Mat* mask) const;
    // Ограничивает прямоугольник границами кадра.
    cv::Rect2f clip_rect(const cv::Rect2f& rect, const cv::Size& size) const;
    // Вычисляет контраст ROI для фильтрации слабых целей.
    float compute_contrast(const cv::Mat& frame, const cv::Rect2f& roi) const;
    // Проверяет попадание точки в прямоугольник с паддингом.
    bool point_in_rect_with_padding(const cv::Rect2f& rect, int x, int y, int pad) const;

    cv::Mat flood_fill_overlay_; // - flood_fill_overlay_: визуальный оверлей зоны floodfill.
    cv::Mat flood_fill_mask_; // - flood_fill_mask_: маска зоны floodfill.
    long long overlay_expire_ms_ = 0; // - overlay_expire_ms_: таймер жизни оверлея.
};
