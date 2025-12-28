#include "detector.h"
#include <algorithm>
#include <cmath>
#include <iostream>

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
        cfg_.diff_threshold = read_required<int>(*detector, "diff_threshold");
        cfg_.chroma_threshold = read_required<int>(*detector, "chroma_threshold");
        cfg_.min_area = read_required<int>(*detector, "min_area");
        cfg_.sensitivity = read_required<double>(*detector, "sensitivity");
        cfg_.morph_kernel = read_required<int>(*detector, "morph_kernel");
        cfg_.downscale = read_required<double>(*detector, "downscale");

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
                        dnn_model_.emplace(cfg_.dnn_model_path);
                    } else {
                        dnn_model_.emplace(cfg_.dnn_model_path, cfg_.dnn_config_path);
                    }
                    dnn_model_->setInputParams(
                            cfg_.dnn_scale,
                            cv::Size(cfg_.dnn_input_width, cfg_.dnn_input_height),
                            cfg_.dnn_mean,
                            cfg_.dnn_swap_rb
                    );
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
                      << " (falling back to motion detector). "
                      << "The configured model may be incompatible with OpenCV "
                      << "DetectionModel in 4.5.4; try a supported model "
                      << "(e.g. SSD/YOLOv3/YOLOv4) or implement custom postprocess."
                      << std::endl;
            std::cerr.flush();
            dnn_ready_ = false;
            cfg_.use_dnn = false;
        }
    }
    return detect_motion(frame_bgr);
}

std::vector<cv::Rect2f> Detector::detect_dnn(const cv::Mat& frame_bgr) {
    std::vector<cv::Rect2f> out;
    if (frame_bgr.empty()) return out;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    if (!dnn_model_) {
        return out;
    }
    dnn_model_->detect(frame_bgr, class_ids, confidences, boxes,
                       cfg_.dnn_conf_threshold, cfg_.dnn_nms_threshold);

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (cfg_.dnn_class_id >= 0 && class_ids[i] != cfg_.dnn_class_id) {
            continue;
        }
        const cv::Rect &r = boxes[i];
        out.emplace_back(static_cast<float>(r.x),
                         static_cast<float>(r.y),
                         static_cast<float>(r.width),
                         static_cast<float>(r.height));
    }
    return out;
}

std::vector<cv::Rect2f> Detector::detect_motion(const cv::Mat& frame_bgr) {
    std::vector<cv::Rect2f> out;
    if (frame_bgr.empty()) return out;

    cv::Mat frame = frame_bgr;
    cv::Mat small;
    double s = cfg_.downscale;
    if (s > 0.0 && s < 1.0) {
        cv::resize(frame_bgr, small, cv::Size(), s, s, cv::INTER_AREA);
        frame = small;
    }

    cv::Mat ycrcb;
    cv::cvtColor(frame, ycrcb, cv::COLOR_BGR2YCrCb);

    if (prev_ycrcb_.empty()) {
        prev_ycrcb_ = ycrcb;
        return out;
    }

    cv::Mat diff;
    cv::absdiff(ycrcb, prev_ycrcb_, diff);
    prev_ycrcb_ = ycrcb;

    cv::Mat thr;
    const double sensitivity = cfg_.sensitivity > 0.0 ? cfg_.sensitivity : 1.0;
    const int diff_threshold = std::max(0, static_cast<int>(std::lround(cfg_.diff_threshold * sensitivity)));
    const int chroma_threshold = std::max(1, static_cast<int>(std::lround(cfg_.chroma_threshold * sensitivity)));
    const int min_area = std::max(1, static_cast<int>(std::lround(cfg_.min_area * sensitivity)));
    cv::Mat diff_channels[3];
    cv::split(diff, diff_channels);
    cv::Mat chroma_diff;
    cv::max(diff_channels[1], diff_channels[2], chroma_diff);
    cv::threshold(chroma_diff, thr, chroma_threshold, 255, cv::THRESH_BINARY);

    if (diff_threshold > 0) {
        cv::Mat luma_thr;
        cv::threshold(diff_channels[0], luma_thr, diff_threshold, 255, cv::THRESH_BINARY);
        cv::bitwise_or(thr, luma_thr, thr);
    }

    if (cfg_.morph_kernel > 0) {
        cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(cfg_.morph_kernel, cfg_.morph_kernel));
        cv::morphologyEx(thr, thr, cv::MORPH_CLOSE, k);
        cv::morphologyEx(thr, thr, cv::MORPH_OPEN, k);
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thr, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (auto& c : contours) {
        double area = cv::contourArea(c);
        if (area < min_area) continue;
        cv::Rect r = cv::boundingRect(c);

        cv::Rect2f rf((float)r.x, (float)r.y, (float)r.width, (float)r.height);
        if (s > 0.0 && s < 1.0) {
            rf.x /= (float)s; rf.y /= (float)s; rf.width /= (float)s; rf.height /= (float)s;
        }
        out.push_back(rf);
    }
    return out;
}
