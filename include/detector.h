#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include "config.h"

class Detector {
private:
    struct DetectorConfig {
        // Минимальная разница яркости (0 = отключить яркостной анализ)
        int diff_threshold = 0;

        // Минимальная разница цветности (Cb/Cr) между кадрами
        int chroma_threshold = 10;

        // Минимальная площадь bbox
        int min_area = 20;

        // Коэффициент чувствительности детектора
        double sensitivity = 1.0;

        // Размер морфологического ядра
        int morph_kernel = 3;

        // Downscale перед детекцией
        double downscale = 1.0;

        // Использовать DNN детектор вместо motion
        bool use_dnn = false;

        // Путь к модели DNN (например, ONNX)
        std::string dnn_model_path;

        // Путь к конфигу (если требуется для конкретного фреймворка)
        std::string dnn_config_path;

        // Размер входа сети
        int dnn_input_width = 640;
        int dnn_input_height = 640;

        // Нормализация входа
        float dnn_scale = 1.0f / 255.0f;
        cv::Scalar dnn_mean = cv::Scalar(0, 0, 0);
        bool dnn_swap_rb = true;

        // Пороги детекции
        float dnn_conf_threshold = 0.35f;
        float dnn_nms_threshold = 0.45f;

        // Фильтр по классу (-1 = все классы)
        int dnn_class_id = -1;
    };

public:
    explicit Detector(const toml::table& tbl);

    // Returns detections in input frame coordinates (BGR)
    std::vector<cv::Rect2f> detect(const cv::Mat& frame_bgr);

private:
    DetectorConfig cfg_;
    cv::Mat prev_ycrcb_;
    std::optional<cv::dnn::DetectionModel> dnn_model_;
    bool dnn_ready_ = false;

    bool load_detector_config(const toml::table& tbl);
    std::vector<cv::Rect2f> detect_motion(const cv::Mat& frame_bgr);
    std::vector<cv::Rect2f> detect_dnn(const cv::Mat& frame_bgr);
};
