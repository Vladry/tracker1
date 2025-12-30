#include "yolo_postprocess.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/dnn.hpp>

namespace {
    size_t tensor_elem_count(const rknn_tensor_attr &attr) {
        size_t count = 1;
        for (uint32_t i = 0; i < attr.n_dims; ++i) {
            count *= static_cast<size_t>(attr.dims[i]);
        }
        return count;
    }
}

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
) {
    std::vector<cv::Rect2f> out;
    if (output.empty()) {
        std::cout << "[POST] empty output матрица" << std::endl;
        return out;
    }

    cv::Mat dets;
    if (output.dims == 3) {
        const int dim1 = output.size[1];
        const int dim2 = output.size[2];
        if (dim1 <= 0 || dim2 <= 0) {
            std::cout << "[POST] некорректные dims: " << dim1 << "x" << dim2 << std::endl;
            return out;
        }
        cv::Mat data(dim1, dim2, CV_32F, const_cast<float*>(output.ptr<float>()));
        dets = (dim1 < dim2) ? data.t() : data;
    } else if (output.dims == 2) {
        dets = output;
    } else {
        std::cout << "[POST] неподдерживаемое число измерений: " << output.dims << std::endl;
        return out;
    }

    const int num_attrs = dets.cols;
    if (num_attrs < 6) {
        std::cout << "[POST] слишком мало атрибутов: " << num_attrs << std::endl;
        return out;
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;

    if (scale <= 0.0f) {
        std::cout << "[POST] некорректный scale: " << scale << std::endl;
        return out;
    }
    std::cout << "[POST] dets: rows=" << dets.rows
              << " cols=" << dets.cols
              << " scale=" << scale
              << " pad=(" << pad_x << "," << pad_y << ")"
              << std::endl;

    auto sigmoid = [](float v) {
        return 1.0f / (1.0f + std::exp(-v));
    };

    const bool has_objectness = (num_attrs - 4) > 1 && num_attrs != 84;
    std::cout << "[POST] layout: "
              << (has_objectness ? "obj+class" : "class-only")
              << " attrs=" << num_attrs
              << std::endl;

    float max_obj = 0.0f;
    float max_class = 0.0f;
    float max_score = 0.0f;
    int passed_threshold = 0;

    for (int i = 0; i < dets.rows; ++i) {
        const float* row = dets.ptr<float>(i);
        float x = row[0];
        float y = row[1];
        float w = row[2];
        float h = row[3];

        bool needs_sigmoid = false;
        for (int c = 4; c < num_attrs; ++c) {
            if (row[c] < 0.0f || row[c] > 1.0f) {
                needs_sigmoid = true;
                break;
            }
        }

        int best_class = -1;
        float best_score = 0.0f;
        float best_class_score = 0.0f;
        float obj_score = 1.0f;
        int class_offset = 4;
        if (has_objectness) {
            obj_score = row[4];
            if (needs_sigmoid) {
                obj_score = sigmoid(obj_score);
            }
            if (obj_score > max_obj) {
                max_obj = obj_score;
            }
            class_offset = 5;
        }

        for (int c = class_offset; c < num_attrs; ++c) {
            float class_score = needs_sigmoid ? sigmoid(row[c]) : row[c];
            float score = class_score * obj_score;
            if (score > best_score) {
                best_score = score;
                best_class = c - class_offset;
                best_class_score = class_score;
            }
        }
        if (best_class_score > max_class) {
            max_class = best_class_score;
        }
        if (best_score > max_score) {
            max_score = best_score;
        }

        if (best_score < conf_threshold) {
            continue;
        }
        if (class_id >= 0 && best_class != class_id) {
            continue;
        }
        passed_threshold++;

        if (x <= 1.0f && y <= 1.0f && w <= 1.0f && h <= 1.0f) {
            x *= static_cast<float>(input_size.width);
            y *= static_cast<float>(input_size.height);
            w *= static_cast<float>(input_size.width);
            h *= static_cast<float>(input_size.height);
        }

        float left = x - 0.5f * w;
        float top = y - 0.5f * h;
        float width = w;
        float height = h;

        left -= pad_x;
        top -= pad_y;

        left /= scale;
        top /= scale;
        width /= scale;
        height /= scale;

        float right = left + width;
        float bottom = top + height;
        left = std::max(0.0f, std::min(left, static_cast<float>(frame_size.width - 1)));
        top = std::max(0.0f, std::min(top, static_cast<float>(frame_size.height - 1)));
        right = std::max(0.0f, std::min(right, static_cast<float>(frame_size.width)));
        bottom = std::max(0.0f, std::min(bottom, static_cast<float>(frame_size.height)));
        width = right - left;
        height = bottom - top;
        if (width <= 1.0f || height <= 1.0f) {
            continue;
        }

        boxes.emplace_back(static_cast<int>(left),
                           static_cast<int>(top),
                           static_cast<int>(width),
                           static_cast<int>(height));
        confidences.push_back(best_score);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);
    std::cout << "[POST] boxes=" << boxes.size()
              << " after_nms=" << indices.size()
              << " passed_th=" << passed_threshold
              << " max_obj=" << max_obj
              << " max_class=" << max_class
              << " max_score=" << max_score
              << " conf_th=" << conf_threshold
              << " nms_th=" << nms_threshold
              << std::endl;

    for (int idx : indices) {
        const cv::Rect &r = boxes[idx];
        out.emplace_back(static_cast<float>(r.x),
                         static_cast<float>(r.y),
                         static_cast<float>(r.width),
                         static_cast<float>(r.height));
    }

    return out;
}

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
) {
    if (outputs.empty() || output_attrs.empty()) {
        return {};
    }

    int best_idx = 0;
    size_t best_count = 0;
    for (size_t i = 0; i < output_attrs.size(); ++i) {
        const size_t count = tensor_elem_count(output_attrs[i]);
        if (count > best_count) {
            best_count = count;
            best_idx = static_cast<int>(i);
        }
    }

    const rknn_tensor_attr &out_attr = output_attrs[static_cast<size_t>(best_idx)];
    std::vector<int> sizes(out_attr.n_dims);
    for (uint32_t i = 0; i < out_attr.n_dims; ++i) {
        sizes[i] = static_cast<int>(out_attr.dims[i]);
    }

    cv::Mat output_mat(out_attr.n_dims, sizes.data(), CV_32F, outputs[best_idx].buf);
    return decode_yolov8_output(
            output_mat,
            input_size,
            frame_size,
            scale,
            static_cast<float>(pad_left),
            static_cast<float>(pad_top),
            conf_threshold,
            nms_threshold,
            class_id
    );
}
