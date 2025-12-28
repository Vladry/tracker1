#include "yolo_postprocess.h"
#include <opencv2/dnn.hpp>

std::vector<cv::Rect2f> decode_yolov8_output(
        const cv::Mat& output,
        const cv::Size& input_size,
        const cv::Size& frame_size,
        float conf_threshold,
        float nms_threshold,
        int class_id
) {
    std::vector<cv::Rect2f> out;
    if (output.empty()) {
        return out;
    }

    cv::Mat dets;
    if (output.dims == 3) {
        const int dim1 = output.size[1];
        const int dim2 = output.size[2];
        if (dim1 <= 0 || dim2 <= 0) {
            return out;
        }
        cv::Mat data(dim1, dim2, CV_32F, const_cast<float*>(output.ptr<float>()));
        dets = (dim1 < dim2) ? data.t() : data;
    } else if (output.dims == 2) {
        dets = output;
    } else {
        return out;
    }

    const int num_attrs = dets.cols;
    if (num_attrs < 6) {
        return out;
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;

    const float scale_x = static_cast<float>(frame_size.width) / static_cast<float>(input_size.width);
    const float scale_y = static_cast<float>(frame_size.height) / static_cast<float>(input_size.height);

    for (int i = 0; i < dets.rows; ++i) {
        const float* row = dets.ptr<float>(i);
        float x = row[0];
        float y = row[1];
        float w = row[2];
        float h = row[3];

        int best_class = -1;
        float best_score = 0.0f;
        for (int c = 4; c < num_attrs; ++c) {
            if (row[c] > best_score) {
                best_score = row[c];
                best_class = c - 4;
            }
        }

        if (best_score < conf_threshold) {
            continue;
        }
        if (class_id >= 0 && best_class != class_id) {
            continue;
        }

        if (x <= 1.0f && y <= 1.0f && w <= 1.0f && h <= 1.0f) {
            x *= static_cast<float>(input_size.width);
            y *= static_cast<float>(input_size.height);
            w *= static_cast<float>(input_size.width);
            h *= static_cast<float>(input_size.height);
        }

        float left = (x - 0.5f * w) * scale_x;
        float top = (y - 0.5f * h) * scale_y;
        float width = w * scale_x;
        float height = h * scale_y;

        boxes.emplace_back(static_cast<int>(left),
                           static_cast<int>(top),
                           static_cast<int>(width),
                           static_cast<int>(height));
        confidences.push_back(best_score);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);

    for (int idx : indices) {
        const cv::Rect &r = boxes[idx];
        out.emplace_back(static_cast<float>(r.x),
                         static_cast<float>(r.y),
                         static_cast<float>(r.width),
                         static_cast<float>(r.height));
    }

    return out;
}
