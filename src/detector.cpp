#include "detector.h"
#include <iostream>
#include "yolo_postprocess.h"

Detector::Detector(const toml::table& tbl) {
    load_detector_config(tbl);
}

bool Detector::load_detector_config(const toml::table& tbl) {
    // --------------------------- [detector] ---------------------------
    try {
        const auto *detector = tbl["detector"].as_table();
        if (!detector) {
            throw std::runtime_error("missing [detector] table");
        }
        if (const auto node = detector->get("use_dnn"); node && node->value<bool>()) {
            cfg_.use_dnn = *node->value<bool>(); std::cout<<"use_dnn= "<<cfg_.use_dnn<<std::endl;
            std::cout.flush();
        }
        if (const auto node = detector->get("dnn_model_path"); node && node->value<std::string>()) {
            cfg_.dnn_model_path = *node->value<std::string>();std::cout<<"dnn_model_path= "<<cfg_.dnn_model_path<<std::endl;
            std::cout.flush();
        }
        if (const auto node = detector->get("dnn_config_path"); node && node->value<std::string>()) {
            cfg_.dnn_config_path = *node->value<std::string>();
        }
        if (const auto node = detector->get("dnn_input_width"); node && node->value<int>()) {
            cfg_.dnn_input_width = *node->value<int>();
        }
        if (const auto node = detector->get("dnn_input_height"); node && node->value<int>()) {
            cfg_.dnn_input_height = *node->value<int>();
        }
        if (const auto node = detector->get("dnn_scale"); node && node->value<double>()) {
            cfg_.dnn_scale = static_cast<float>(*node->value<double>());
        }
        if (const auto node = detector->get("dnn_swap_rb"); node && node->value<bool>()) {
            cfg_.dnn_swap_rb = *node->value<bool>();
        }
        if (const auto node = detector->get("dnn_conf_threshold"); node && node->value<double>()) {
            cfg_.dnn_conf_threshold = static_cast<float>(*node->value<double>());
        }
        if (const auto node = detector->get("dnn_nms_threshold"); node && node->value<double>()) {
            cfg_.dnn_nms_threshold = static_cast<float>(*node->value<double>());
        }
        if (const auto node = detector->get("dnn_class_id"); node && node->value<int>()) {
            cfg_.dnn_class_id = *node->value<int>();
        }

        if (cfg_.use_dnn) {
            if (cfg_.dnn_model_path.empty()) {
                std::cerr << "detector config: dnn_model_path is empty, fallback to motion detector"
                          << std::endl;
                std::cerr.flush();
                cfg_.use_dnn = false;
            } else {
                try {
                    if (cfg_.dnn_config_path.empty()) {
                        dnn_net_ = cv::dnn::readNet(cfg_.dnn_model_path);
                    } else {
                        dnn_net_ = cv::dnn::readNet(cfg_.dnn_model_path, cfg_.dnn_config_path);
                    }
                    dnn_output_names_ = dnn_net_.getUnconnectedOutLayersNames();
                    dnn_ready_ = true;
                } catch (const std::exception &e) {
                    std::cerr << "detector dnn init failed: " << e.what() << std::endl;
                    std::cerr.flush();
                    dnn_ready_ = false;
                    cfg_.use_dnn = false;
                }
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
    if (cfg_.use_dnn && dnn_ready_) {
        try {
            return detect_dnn(frame_bgr);
        } catch (const cv::Exception &e) {
            std::cerr << "detector dnn detect failed: " << e.what()
                      << " (disabling DNN for this session)."
                      << std::endl;
            std::cerr.flush();
            dnn_ready_ = false;
            cfg_.use_dnn = false;
        }
    }
    return {};
}

std::vector<cv::Rect2f> Detector::detect_dnn(const cv::Mat& frame_bgr) {
    std::vector<cv::Rect2f> out;
    if (frame_bgr.empty()) return out;

    const cv::Size input_size(cfg_.dnn_input_width, cfg_.dnn_input_height);
    cv::Mat blob = cv::dnn::blobFromImage(
            frame_bgr,
            cfg_.dnn_scale,
            input_size,
            cfg_.dnn_mean,
            cfg_.dnn_swap_rb,
            false
    );

    dnn_net_.setInput(blob);
    std::vector<cv::Mat> outputs;
    dnn_net_.forward(outputs, dnn_output_names_);
    if (outputs.empty()) {
        return out;
    }

    return decode_yolov8_output(
            outputs[0],
            input_size,
            frame_bgr.size(),
            cfg_.dnn_conf_threshold,
            cfg_.dnn_nms_threshold,
            cfg_.dnn_class_id
    );
}
