#include "util/offset_calculator.h"

OffsetResult OffsetCalculator::compute(const cv::Size& frameSize, const cv::Rect2f& bbox) const {
    OffsetResult r;
    double cx = bbox.x + bbox.width * 0.5;
    double cy = bbox.y + bbox.height * 0.5;
    r.dx_px = cx - (frameSize.width * 0.5);
    r.dy_px = cy - (frameSize.height * 0.5);
    return r;
}
