#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

std::vector<cv::Rect2f> decode_yolov8_output(
        const cv::Mat& output,
        const cv::Size& input_size,
        const cv::Size& frame_size,
        float scale,
        float pad_x,
        float pad_y,
        float conf_threshold,
        float nms_threshold,
        int class_id
);
