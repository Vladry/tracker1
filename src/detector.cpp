#include "detector.h"
#include <cstring>
#include <fstream>
#include <iostream>
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

std::vector<cv::Rect2f> Detector::detect_rknn(const cv::Mat& frame_bgr) {
    std::vector<cv::Rect2f> out;
    if (frame_bgr.empty()) return out;

    const cv::Size input_size = input_size_from_attr(input_attr_);
    if (input_size.width <= 0 || input_size.height <= 0) {
        return out;
    }

    cv::Mat resized;
    cv::resize(frame_bgr, resized, input_size);
    if (cfg_.rknn_swap_rb) {
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    }

    rknn_input input{};
    std::vector<uint8_t> input_u8;
    std::vector<float> input_f32;

    if (input_attr_.type == RKNN_TENSOR_UINT8) {
        if (is_nchw(input_attr_)) {
            std::vector<cv::Mat> channels;
            cv::split(resized, channels);
            input_u8.resize(static_cast<size_t>(resized.total() * resized.channels()));
            size_t offset = 0;
            for (const auto &ch : channels) {
                const size_t len = static_cast<size_t>(ch.total());
                std::memcpy(input_u8.data() + offset, ch.data, len);
                offset += len;
            }
            input.buf = input_u8.data();
            input.size = input_u8.size();
        } else {
            input.buf = resized.data;
            input.size = static_cast<size_t>(resized.total() * resized.elemSize());
        }
        input.type = RKNN_TENSOR_UINT8;
    } else {
        cv::Mat float_img;
        resized.convertTo(float_img, CV_32F);
        if (cfg_.rknn_mean != cv::Scalar(0, 0, 0)) {
            float_img -= cfg_.rknn_mean;
        }
        if (cfg_.rknn_scale != 1.0f) {
            float_img *= cfg_.rknn_scale;
        }

        if (is_nchw(input_attr_)) {
            std::vector<cv::Mat> channels;
            cv::split(float_img, channels);
            input_f32.resize(static_cast<size_t>(float_img.total() * float_img.channels()));
            size_t offset = 0;
            for (const auto &ch : channels) {
                const size_t len = static_cast<size_t>(ch.total());
                std::memcpy(input_f32.data() + offset, ch.ptr<float>(), len * sizeof(float));
                offset += len;
            }
            input.buf = input_f32.data();
            input.size = input_f32.size() * sizeof(float);
        } else {
            input.buf = float_img.data;
            input.size = static_cast<size_t>(float_img.total() * float_img.elemSize());
        }
        input.type = RKNN_TENSOR_FLOAT32;
    }

    input.index = 0;
    input.fmt = input_attr_.fmt;
    input.pass_through = 0;

    if (rknn_inputs_set(rknn_ctx_, 1, &input) != RKNN_SUCC) {
        throw std::runtime_error("rknn_inputs_set failed");
    }
    if (rknn_run(rknn_ctx_, nullptr) != RKNN_SUCC) {
        throw std::runtime_error("rknn_run failed");
    }

    const int output_count = static_cast<int>(output_attrs_.size());
    if (output_count <= 0) {
        return out;
    }

    std::vector<rknn_output> outputs(static_cast<size_t>(output_count));
    for (int i = 0; i < output_count; ++i) {
        outputs[i].want_float = 1;
        outputs[i].is_prealloc = 0;
    }

    if (rknn_outputs_get(rknn_ctx_, output_count, outputs.data(), nullptr) != RKNN_SUCC) {
        throw std::runtime_error("rknn_outputs_get failed");
    }

    const rknn_tensor_attr &out_attr = output_attrs_[0];
    std::vector<int> sizes(out_attr.n_dims);
    for (uint32_t i = 0; i < out_attr.n_dims; ++i) {
        sizes[i] = static_cast<int>(out_attr.dims[i]);
    }

    cv::Mat output_mat(out_attr.n_dims, sizes.data(), CV_32F, outputs[0].buf);
    out = decode_yolov8_output(
            output_mat,
            input_size,
            frame_bgr.size(),
            cfg_.rknn_conf_threshold,
            cfg_.rknn_nms_threshold,
            cfg_.rknn_class_id
    );

    rknn_outputs_release(rknn_ctx_, output_count, outputs.data());
    return out;
}
