#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "config.h"

class MotionDetector {
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
    };

public:
    explicit MotionDetector(const toml::table& tbl);

    // Returns detections in input frame coordinates (BGR)
    std::vector<cv::Rect2f> detect(const cv::Mat& frame_bgr);

private:
    DetectorConfig cfg_;
    cv::Mat prev_ycrcb_;

    bool load_detector_config(const toml::table& tbl);
};

