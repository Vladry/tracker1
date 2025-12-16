#pragma once
#include <opencv2/opencv.hpp>
#include <mutex>
#include <condition_variable>
#include <atomic>

class FrameStore {
public:
    FrameStore() = default;

    void setFrame(cv::Mat&& frame);                 // используется RtspWorker
    void pushFrame(const cv::Mat& frame);           // опционально
    bool waitFrame(cv::Mat& out, int timeout_ms);
    void stop();

private:
    cv::Mat last_;
    bool has_frame_ = false;

    std::mutex m_;
    std::condition_variable cv_;
    std::atomic<bool> stop_{false};
};
