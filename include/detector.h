#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "config.h"
#include <toml++/toml.h>   // ОБЯЗАТЕЛЬНО, forward-decl НЕЛЬЗЯ

class MotionDetector {
private:
    struct DetectorConfig {
        // Минимальная разница яркости
        int diff_threshold = 20;

        // Минимальная площадь bbox
        int min_area = 10;

        // Размер морфологического ядра
        int morph_kernel = 3;

        // Downscale перед детекцией
        double downscale = 1.0;
    };

    //    explicit MotionDetector(DetectorConfig dcfg);
    explicit MotionDetector(const toml::table& tbl);

    // Returns detections in input frame coordinates (BGR)
    std::vector<cv::Rect2f> detect(const cv::Mat& frame_bgr);

private:
    DetectorConfig cfg_;
    cv::Mat prev_gray_;

    bool load_detector_config(const toml::table& tbl);
};

