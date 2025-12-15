#pragma once
#include <opencv2/core.hpp>
#include <vector>

namespace blob {

struct MergeParams {
    float iou_threshold  = 0.12f; // merge if IoU >= threshold
    int   gap_px         = 10;    // merge if rects are close (even with low IoU)
    float expand_factor  = 0.08f; // expand rects before testing
};

// Merge fragmented detections into larger blobs.
// Designed to reduce "one body -> many sub-blobs" effect from motion-diff detector.
std::vector<cv::Rect2f> merge_blobs(const std::vector<cv::Rect2f>& in,
                                   const cv::Size& frame_size,
                                   const MergeParams& p);

} // namespace blob
