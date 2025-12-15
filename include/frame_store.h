#pragma once
#include <opencv2/opencv.hpp>
#include <mutex>
#include <atomic>

class FrameStore {
public:
    FrameStore() = default;

    void setFrame(cv::Mat frame_bgr);
    bool tryGetFrame(cv::Mat& out_bgr);

private:
    std::mutex mtx_;
    cv::Mat last_;
    std::atomic<bool> has_{false};
};
