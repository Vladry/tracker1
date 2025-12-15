#include "core/frame_store.h"

void FrameStore::setFrame(cv::Mat frame_bgr) {
    std::lock_guard<std::mutex> lock(mtx_);
    last_ = std::move(frame_bgr);
    has_.store(true, std::memory_order_release);
}

bool FrameStore::tryGetFrame(cv::Mat& out_bgr) {
    if (!has_.load(std::memory_order_acquire)) return false;
    std::lock_guard<std::mutex> lock(mtx_);
    if (last_.empty()) return false;
    out_bgr = last_.clone();
    return true;
}
