#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace blob {

    struct MergeParams {
        // existing fields
        float iou_threshold = 0.12f;
        int   gap_px        = 10;
        float expand_factor = 0.08f;

        // new (agglomeration)
        float center_dist_px = 80.0f;   // merge if centers close
        int   min_area_px    = 250;     // filter small
        int   max_out        = 20;      // keep top-N by area
    };

    std::vector<cv::Rect2f> merge_blobs(const std::vector<cv::Rect2f>& in,
                                        const cv::Size& frame_size,
                                        const MergeParams& p);

} // namespace blob
