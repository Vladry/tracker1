#pragma once

#include <opencv2/opencv.hpp>
#include <mutex>
#include <string>
#include <vector>
#include "config.h"

struct StaticTarget {
    int id = -1; // - идентификатор статической цели.
    cv::Rect2f bbox; // - прямоугольник статической цели.
};

class StaticTargetManager {
public:
    struct Config {
        int MAX_TARGETS = 5; // - максимум статических целей.
        int CLICK_PADDING = 6; // - паддинг вокруг ROI клика.
        int REMOVE_PADDING = 6; // - паддинг удаления цели по клику.
        int FALLBACK_BOX_SIZE = 40; // - размер запасного bbox при ошибке floodfill.
        float MAX_AREA_RATIO = 0.1f; // - максимум площади ROI от площади кадра.
        bool CLICK_EQUALIZE = true; // - включение эквализации гистограммы перед floodfill.
        int FLOODFILL_LO_DIFF = 20; // - нижний порог flood fill.
        int FLOODFILL_HI_DIFF = 20; // - верхний порог flood fill.
        int OVERLAY_TTL_SECONDS = 3; // - время жизни оверлея floodfill.
        float FLOODFILL_OVERLAY_ALPHA = 0.7f; // - альфа заливки оверлея.
        int MIN_AREA = 60; // - минимальная площадь ROI.
        int MIN_WIDTH = 6; // - минимальная ширина ROI.
        int MIN_HEIGHT = 6; // - минимальная высота ROI.
        float MIN_CONTRAST = 5.0f; // - минимальный контраст ROI.
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
    Config cfg_; // - настройки статического детектора.
    LoggingConfig log_cfg_; // - настройки логирования.
    std::vector<StaticTarget> targets_; // - список статических целей.
    int next_id_ = 1; // - счётчик идентификаторов целей.
    std::mutex mutex_; // - защита доступа из разных потоков.

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

    cv::Mat flood_fill_overlay_; // - визуальный оверлей зоны floodfill.
    cv::Mat flood_fill_mask_; // - маска зоны floodfill.
    long long overlay_expire_ms_ = 0; // - таймер жизни оверлея.
};
