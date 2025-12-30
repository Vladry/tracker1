#pragma once
#include <opencv2/opencv.hpp>
#include <rknn_api.h>
#include <vector>
#include "config.h"

class Detector {
private:
    struct DetectorConfig {
        // Использовать детектор движения
        bool use_motion = true;

        // Использовать RKNN детектор
        bool use_rknn = false;

        // Путь к модели RKNN
        std::string rknn_model_path;

        // Нормализация входа
        float rknn_scale = 1.0f / 255.0f;
        cv::Scalar rknn_mean = cv::Scalar(0, 0, 0);
        bool rknn_swap_rb = true;

        // Пороги детекции
        float rknn_conf_threshold = 0.35f;
        float rknn_nms_threshold = 0.45f;

        // Фильтр по классу (-1 = все классы)
        int rknn_class_id = -1;

        // -------------------- motion detector --------------------
        int motion_history = 500;
        double motion_var_threshold = 16.0;
        bool motion_detect_shadows = false;
        int motion_min_area = 500;
        int motion_min_width = 10;
        int motion_min_height = 10;
        int motion_blur_size = 5;
        int motion_morph_size = 3;
        int motion_erode_iterations = 1;
        int motion_dilate_iterations = 2;
    };

public:
    explicit Detector(const toml::table& tbl);
    ~Detector();

    // Returns detections in input frame coordinates (BGR)
    std::vector<cv::Rect2f> detect(const cv::Mat& frame_bgr);

private:
    DetectorConfig cfg_;
    rknn_context rknn_ctx_ = 0;
    rknn_tensor_attr input_attr_{};
    std::vector<rknn_tensor_attr> output_attrs_;
    bool rknn_ready_ = false;
    cv::Ptr<cv::BackgroundSubtractor> motion_subtractor_;

    bool load_detector_config(const toml::table& tbl);
    std::vector<cv::Rect2f> detect_rknn(const cv::Mat& frame_bgr);
    std::vector<cv::Rect2f> detect_motion(const cv::Mat& frame_bgr);
};
