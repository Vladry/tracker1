#pragma once

#include <opencv2/opencv.hpp>
#include <rknn_api.h>
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

std::vector<cv::Rect2f> yolo_postprocess(
        const std::vector<rknn_output>& outputs,
        const std::vector<rknn_tensor_attr>& output_attrs,
        float conf_threshold,
        float nms_threshold,
        int class_id,
        const cv::Size& input_size,
        const cv::Size& frame_size,
        int pad_left,
        int pad_top,
        float scale
);
