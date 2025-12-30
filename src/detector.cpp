#include "detector.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include "yolo_postprocess.h"

namespace {
    cv::Size input_size_from_attr(const rknn_tensor_attr &attr) {
        if (attr.n_dims >= 4) {
            if (attr.fmt == RKNN_TENSOR_NCHW) {
                return {static_cast<int>(attr.dims[3]), static_cast<int>(attr.dims[2])};
            }
            return {static_cast<int>(attr.dims[2]), static_cast<int>(attr.dims[1])};
        }
        return {0, 0};
    }

    bool is_nchw(const rknn_tensor_attr &attr) {
        return attr.fmt == RKNN_TENSOR_NCHW;
    }

    size_t tensor_elem_count(const rknn_tensor_attr &attr) {
        size_t count = 1;
        for (uint32_t i = 0; i < attr.n_dims; ++i) {
            count *= static_cast<size_t>(attr.dims[i]);
        }
        return count;
    }

    std::string tensor_dims_to_string(const rknn_tensor_attr &attr) {
        std::ostringstream oss;
        oss << "[";
        for (uint32_t i = 0; i < attr.n_dims; ++i) {
            oss << attr.dims[i];
            if (i + 1 < attr.n_dims) {
                oss << "x";
            }
        }
        oss << "]";
        return oss.str();
    }
}

Detector::Detector(const toml::table& tbl) {
    load_detector_config(tbl);
}

Detector::~Detector() {
    if (rknn_ctx_ != 0) {
        rknn_destroy(rknn_ctx_);
        rknn_ctx_ = 0;
    }
}

bool Detector::load_detector_config(const toml::table& tbl) {
    // --------------------------- [detector] ---------------------------
    try {
        const auto *detector = tbl["detector"].as_table();
        if (!detector) {
            throw std::runtime_error("missing [detector] table");
        }
        if (const auto node = detector->get("use_motion"); node && node->value<bool>()) {
            cfg_.use_motion = *node->value<bool>();
        }
        if (const auto node = detector->get("use_rknn"); node && node->value<bool>()) {
            cfg_.use_rknn = *node->value<bool>();
        }
        if (const auto node = detector->get("rknn_model_path"); node && node->value<std::string>()) {
            cfg_.rknn_model_path = *node->value<std::string>();
        }
        if (const auto node = detector->get("rknn_scale"); node && node->value<double>()) {
            cfg_.rknn_scale = static_cast<float>(*node->value<double>());
        }
        if (const auto node = detector->get("rknn_swap_rb"); node && node->value<bool>()) {
            cfg_.rknn_swap_rb = *node->value<bool>();
        }
        if (const auto node = detector->get("rknn_conf_threshold"); node && node->value<double>()) {
            cfg_.rknn_conf_threshold = static_cast<float>(*node->value<double>());
        }
        if (const auto node = detector->get("rknn_nms_threshold"); node && node->value<double>()) {
            cfg_.rknn_nms_threshold = static_cast<float>(*node->value<double>());
        }
        if (const auto node = detector->get("rknn_class_id"); node && node->value<int>()) {
            cfg_.rknn_class_id = *node->value<int>();
        }

        // --------------------------- [motion_detector] ---------------------------
        if (const auto *motion = tbl["motion_detector"].as_table(); motion) {
            if (const auto node = motion->get("history"); node && node->value<int>()) {
                cfg_.motion_history = *node->value<int>();
            }
            if (const auto node = motion->get("var_threshold"); node && node->value<double>()) {
                cfg_.motion_var_threshold = *node->value<double>();
            }
            if (const auto node = motion->get("detect_shadows"); node && node->value<bool>()) {
                cfg_.motion_detect_shadows = *node->value<bool>();
            }
            if (const auto node = motion->get("min_area"); node && node->value<int>()) {
                cfg_.motion_min_area = *node->value<int>();
            }
            if (const auto node = motion->get("min_width"); node && node->value<int>()) {
                cfg_.motion_min_width = *node->value<int>();
            }
            if (const auto node = motion->get("min_height"); node && node->value<int>()) {
                cfg_.motion_min_height = *node->value<int>();
            }
            if (const auto node = motion->get("blur_size"); node && node->value<int>()) {
                cfg_.motion_blur_size = *node->value<int>();
            }
            if (const auto node = motion->get("morph_size"); node && node->value<int>()) {
                cfg_.motion_morph_size = *node->value<int>();
            }
            if (const auto node = motion->get("erode_iterations"); node && node->value<int>()) {
                cfg_.motion_erode_iterations = *node->value<int>();
            }
            if (const auto node = motion->get("dilate_iterations"); node && node->value<int>()) {
                cfg_.motion_dilate_iterations = *node->value<int>();
            }
        }

        if (cfg_.use_rknn) {
            if (cfg_.rknn_model_path.empty()) {
                std::cerr << "detector config: rknn_model_path is empty, disabling detector" << std::endl;
                std::cerr.flush();
                cfg_.use_rknn = false;
            } else {
                std::ifstream file(cfg_.rknn_model_path, std::ios::binary | std::ios::ate);
                if (!file) {
                    throw std::runtime_error("failed to open rknn model file");
                }
                const std::streamsize size = file.tellg();
                file.seekg(0, std::ios::beg);
                std::vector<unsigned char> model_data(static_cast<size_t>(size));
                if (!file.read(reinterpret_cast<char*>(model_data.data()), size)) {
                    throw std::runtime_error("failed to read rknn model file");
                }

                const int ret = rknn_init(&rknn_ctx_, model_data.data(), model_data.size(), 0, nullptr);
                if (ret != RKNN_SUCC) {
                    throw std::runtime_error("rknn_init failed");
                }

                rknn_input_output_num io_num{};
                if (rknn_query(rknn_ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)) != RKNN_SUCC) {
                    throw std::runtime_error("rknn_query io num failed");
                }

                input_attr_.index = 0;
                if (rknn_query(rknn_ctx_, RKNN_QUERY_INPUT_ATTR, &input_attr_, sizeof(input_attr_)) != RKNN_SUCC) {
                    throw std::runtime_error("rknn_query input attr failed");
                }

                output_attrs_.resize(io_num.n_output);
                for (uint32_t i = 0; i < io_num.n_output; ++i) {
                    output_attrs_[i].index = i;
                    if (rknn_query(rknn_ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i], sizeof(output_attrs_[i])) != RKNN_SUCC) {
                        throw std::runtime_error("rknn_query output attr failed");
                    }
                    std::cout << "rknn output[" << i << "] dims="
                              << tensor_dims_to_string(output_attrs_[i])
                              << " fmt=" << output_attrs_[i].fmt
                              << " type=" << output_attrs_[i].type
                              << std::endl;
                }

                rknn_ready_ = true;
                std::cout << "rknn model loaded: " << cfg_.rknn_model_path
                          << " (inputs=" << io_num.n_input
                          << ", outputs=" << io_num.n_output << ")"
                          << std::endl;
                std::cout.flush();
            }
        }
        return true;

    } catch (const std::exception &e) {
        std::cerr << "detector config load failed  " << e.what() << std::endl;
        std::cerr.flush();
        return false;
    }
};

std::vector<cv::Rect2f> Detector::detect(const cv::Mat& frame_bgr) {
    if (cfg_.use_motion) {
        return detect_motion(frame_bgr);
    }
    if (cfg_.use_rknn && rknn_ready_) {
        try {
            return detect_rknn(frame_bgr);
        } catch (const std::exception &e) {
            std::cerr << "detector rknn detect failed: " << e.what()
                      << " (disabling RKNN for this session)." << std::endl;
            std::cerr.flush();
            rknn_ready_ = false;
            cfg_.use_rknn = false;
        }
    }
    return {};
}

std::vector<cv::Rect2f> Detector::detect_motion(const cv::Mat& frame_bgr) {
    std::vector<cv::Rect2f> out;
    if (frame_bgr.empty()) {
        return out;
    }

    if (!motion_subtractor_) {
        motion_subtractor_ = cv::createBackgroundSubtractorMOG2(
                cfg_.motion_history,
                cfg_.motion_var_threshold,
                cfg_.motion_detect_shadows
        );
    }

    cv::Mat fgmask;
    motion_subtractor_->apply(frame_bgr, fgmask);

    if (cfg_.motion_blur_size > 1) {
        int ksize = cfg_.motion_blur_size;
        if (ksize % 2 == 0) {
            ksize += 1;
        }
        cv::GaussianBlur(fgmask, fgmask, cv::Size(ksize, ksize), 0);
    }

    cv::threshold(fgmask, fgmask, 200, 255, cv::THRESH_BINARY);

    if (cfg_.motion_morph_size > 0) {
        const int k = cfg_.motion_morph_size;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k));
        if (cfg_.motion_erode_iterations > 0) {
            cv::erode(fgmask, fgmask, kernel, cv::Point(-1, -1), cfg_.motion_erode_iterations);
        }
        if (cfg_.motion_dilate_iterations > 0) {
            cv::dilate(fgmask, fgmask, kernel, cv::Point(-1, -1), cfg_.motion_dilate_iterations);
        }
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(fgmask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    out.reserve(contours.size());
    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < cfg_.motion_min_area) {
            continue;
        }
        cv::Rect rect = cv::boundingRect(contour);
        if (rect.width < cfg_.motion_min_width || rect.height < cfg_.motion_min_height) {
            continue;
        }
        out.emplace_back(rect);
    }
    return out;
}

std::vector<cv::Rect2f> Detector::detect_rknn(const cv::Mat& frame_bgr) {
    std::vector<cv::Rect2f> out;
    if (frame_bgr.empty()) return out;

    const cv::Size input_size = input_size_from_attr(input_attr_);
    if (input_size.width <= 0 || input_size.height <= 0) {
        std::cout << "[DET] некорректный input_size: "
                  << input_size.width << "x" << input_size.height << std::endl;
        return out;
    }

    const float scale = std::min(
            static_cast<float>(input_size.width) / static_cast<float>(frame_bgr.cols),
            static_cast<float>(input_size.height) / static_cast<float>(frame_bgr.rows)
    );
    if (scale <= 0.0f) {
        return out;
    }

    const int resized_w = static_cast<int>(std::round(frame_bgr.cols * scale));
    const int resized_h = static_cast<int>(std::round(frame_bgr.rows * scale));
    const int pad_w = input_size.width - resized_w;
    const int pad_h = input_size.height - resized_h;
    const int pad_left = pad_w / 2;
    const int pad_top = pad_h / 2;
    const int pad_right = pad_w - pad_left;
    const int pad_bottom = pad_h - pad_top;
    std::cout << "[DET] frame=" << frame_bgr.cols << "x" << frame_bgr.rows
              << " input=" << input_size.width << "x" << input_size.height
              << " scale=" << scale
              << " resized=" << resized_w << "x" << resized_h
              << " pad=(" << pad_left << "," << pad_top
              << "," << pad_right << "," << pad_bottom << ")"
              << std::endl;

    cv::Mat resized;
    cv::resize(frame_bgr, resized, cv::Size(resized_w, resized_h));

    cv::Mat input(input_size, frame_bgr.type(), cv::Scalar(0, 0, 0));
    cv::Rect roi(pad_left, pad_top, resized.cols, resized.rows);
    resized.copyTo(input(roi));

    cv::Mat input_rgb;
    if (cfg_.rknn_swap_rb) {
        cv::cvtColor(input, input_rgb, cv::COLOR_BGR2RGB);
    } else {
        input_rgb = input;
    }

    cv::Mat input_float;
    input_rgb.convertTo(input_float, CV_32F, cfg_.rknn_scale);

    rknn_input inputs[1]{};
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_FLOAT32;
    inputs[0].fmt = is_nchw(input_attr_) ? RKNN_TENSOR_NCHW : RKNN_TENSOR_NHWC;
    inputs[0].size = tensor_elem_count(input_attr_) * sizeof(float);
    inputs[0].buf = input_float.data;
    if (rknn_inputs_set(rknn_ctx_, 1, inputs) != RKNN_SUCC) {
        throw std::runtime_error("rknn_inputs_set failed");
    }

    if (rknn_run(rknn_ctx_, nullptr) != RKNN_SUCC) {
        throw std::runtime_error("rknn_run failed");
    }

    std::vector<rknn_output> outputs(output_attrs_.size());
    for (size_t i = 0; i < output_attrs_.size(); ++i) {
        outputs[i].want_float = 1;
    }
    if (rknn_outputs_get(rknn_ctx_, static_cast<uint32_t>(outputs.size()), outputs.data(), nullptr) != RKNN_SUCC) {
        throw std::runtime_error("rknn_outputs_get failed");
    }

    std::vector<cv::Rect2f> results = yolo_postprocess(
            outputs,
            output_attrs_,
            cfg_.rknn_conf_threshold,
            cfg_.rknn_nms_threshold,
            cfg_.rknn_class_id,
            input_size,
            frame_bgr.size(),
            pad_left,
            pad_top,
            scale
    );

    rknn_outputs_release(rknn_ctx_, static_cast<uint32_t>(outputs.size()), outputs.data());
    return results;
}
