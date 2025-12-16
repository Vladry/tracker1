#include "core/frame_store.h"
#include <chrono>

void FrameStore::setFrame(cv::Mat&& frame) {
    {
        std::lock_guard<std::mutex> lk(m_);
        last_ = std::move(frame);
        has_frame_ = true;
    }
    cv_.notify_one();
}

void FrameStore::pushFrame(const cv::Mat& frame) {
    {
        std::lock_guard<std::mutex> lk(m_);
        frame.copyTo(last_);
        has_frame_ = true;
    }
    cv_.notify_one();
}

bool FrameStore::waitFrame(cv::Mat& out, int timeout_ms) {
    std::unique_lock<std::mutex> lk(m_);

    if (!cv_.wait_for(
            lk,
            std::chrono::milliseconds(timeout_ms),
            [&]{ return has_frame_ || stop_.load(std::memory_order_acquire); }))
    {
        return false;
    }

    if (stop_.load(std::memory_order_acquire))
        return false;

    out = std::move(last_);
    has_frame_ = false;
    return !out.empty();
}

void FrameStore::stop() {
    stop_.store(true, std::memory_order_release);
    cv_.notify_all();
}
