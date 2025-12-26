#include "detector.h"
#include <algorithm>
#include <cmath>

MotionDetector::MotionDetector(const toml::table& tbl) {
    load_detector_config(tbl);
}

bool MotionDetector::load_detector_config(const toml::table& tbl) {
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
        return true;

    } catch (const std::exception &e) {
        std::cerr << "detecto config load failed  " << e.what() << std::endl;
        return false;
    }

};

std::vector<cv::Rect2f> MotionDetector::detect(const cv::Mat& frame_bgr) {
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




