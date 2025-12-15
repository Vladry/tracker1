#pragma once
#include <opencv2/opencv.hpp>

struct OffsetResult {
    double dx_px = 0.0;
    double dy_px = 0.0;
};

class OffsetCalculator {
public:
    OffsetResult compute(const cv::Size& frameSize, const cv::Rect2f& bbox) const;
};
