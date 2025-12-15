#include "detect/motion_detector.h"

MotionDetector::MotionDetector(const Config& cfg) : cfg_(cfg) {}

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

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    if (prev_gray_.empty()) {
        prev_gray_ = gray;
        return out;
    }

    cv::Mat diff;
    cv::absdiff(gray, prev_gray_, diff);
    prev_gray_ = gray;

    cv::Mat thr;
    cv::threshold(diff, thr, cfg_.diff_threshold, 255, cv::THRESH_BINARY);

    if (cfg_.morph > 0) {
        cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(cfg_.morph, cfg_.morph));
        cv::morphologyEx(thr, thr, cv::MORPH_CLOSE, k);
        cv::morphologyEx(thr, thr, cv::MORPH_OPEN, k);
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thr, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (auto& c : contours) {
        double area = cv::contourArea(c);
        if (area < cfg_.min_area) continue;
        cv::Rect r = cv::boundingRect(c);

        cv::Rect2f rf((float)r.x, (float)r.y, (float)r.width, (float)r.height);
        if (s > 0.0 && s < 1.0) {
            rf.x /= (float)s; rf.y /= (float)s; rf.width /= (float)s; rf.height /= (float)s;
        }
        out.push_back(rf);
    }
    return out;
}
