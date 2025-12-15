#pragma once
#include <opencv2/core.hpp>

namespace util {

// Clamp rectangle to frame bounds (0..W-1, 0..H-1).
cv::Rect2f clampRect(const cv::Rect2f& r, const cv::Size& frameSize);

// Expand rectangle around its center by factor (e.g. 0.08 expands by 8% per side),
// then clamp to frame bounds.
cv::Rect2f expandAndClamp(const cv::Rect2f& r, const cv::Size& frameSize, float expandFactor);

// Intersection over Union. Returns 0..1.
float iou(const cv::Rect2f& a, const cv::Rect2f& b);

// Euclidean distance between centers (pixels).
float centerDistance(const cv::Rect2f& a, const cv::Rect2f& b);

} // namespace util
