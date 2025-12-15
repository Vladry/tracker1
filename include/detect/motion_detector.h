#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class MotionDetector {
public:
    struct Config {
        int diff_threshold = 25;
        int min_area = 250;
        int morph = 3;
        double downscale = 1.0; // 1.0 = no downscale
    };

    explicit MotionDetector(const Config& cfg);

    // Returns detections in input frame coordinates (BGR)
    std::vector<cv::Rect2f> detect(const cv::Mat& frame_bgr);

private:
    Config cfg_;
    cv::Mat prev_gray_;
};
