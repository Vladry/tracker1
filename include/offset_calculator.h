#pragma once
#include <opencv2/opencv.hpp>

struct OffsetResult {
    double dx_px = 0.0; // - dx_px: смещение по X от центра кадра (пиксели).
    double dy_px = 0.0; // - dy_px: смещение по Y от центра кадра (пиксели).
};

class OffsetCalculator {
public:
    // Вычисляет смещение bbox относительно центра кадра.
    OffsetResult compute(const cv::Size& frameSize, const cv::Rect2f& bbox) const;
};
