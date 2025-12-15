#include "util/rect_utils.h"
#include <algorithm>
#include <cmath>

namespace util {

cv::Rect2f clampRect(const cv::Rect2f& r, const cv::Size& frameSize) {
    float x1 = std::max(0.0f, r.x);
    float y1 = std::max(0.0f, r.y);
    float x2 = std::min((float)frameSize.width,  r.x + r.width);
    float y2 = std::min((float)frameSize.height, r.y + r.height);
    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    return cv::Rect2f(x1, y1, w, h);
}

cv::Rect2f expandAndClamp(const cv::Rect2f& r, const cv::Size& frameSize, float expandFactor) {
    if (expandFactor <= 0.0f) return clampRect(r, frameSize);
    float cx = r.x + r.width  * 0.5f;
    float cy = r.y + r.height * 0.5f;
    float w = r.width  * (1.0f + 2.0f * expandFactor);
    float h = r.height * (1.0f + 2.0f * expandFactor);
    cv::Rect2f e(cx - w*0.5f, cy - h*0.5f, w, h);
    return clampRect(e, frameSize);
}

float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    cv::Rect2f inter = a & b;
    float ia = inter.area();
    float ua = a.area() + b.area() - ia;
    if (ua <= 0.0f) return 0.0f;
    return ia / ua;
}

float centerDistance(const cv::Rect2f& a, const cv::Rect2f& b) {
    float ax = a.x + a.width*0.5f;
    float ay = a.y + a.height*0.5f;
    float bx = b.x + b.width*0.5f;
    float by = b.y + b.height*0.5f;
    float dx = ax - bx;
    float dy = ay - by;
    return std::sqrt(dx*dx + dy*dy);
}

} // namespace util
