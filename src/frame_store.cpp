#include "frame_store.h"

void FrameStore::setFrame(cv::Mat&& frame) {
    {
        std::lock_guard<std::mutex> lk(m_);
        if (stop_) return;
        last_ = std::move(frame);
        has_frame_ = true;
        ++seq_;
    }
    cv_.notify_one();
}

void FrameStore::pushFrame(const cv::Mat& frame) {
    {
        std::lock_guard<std::mutex> lk(m_);
        if (stop_) return;
        last_ = frame; // shallow copy (refcount); если нужно "жёстко" - используйте clone() снаружи
        has_frame_ = true;
        ++seq_;
    }
    cv_.notify_one();
}

bool FrameStore::waitFrame(cv::Mat& out, int timeout_ms) {
    std::unique_lock<std::mutex> lk(m_);
    const uint64_t my_seq = seq_;

    const auto pred = [&]() {
        return stop_ || (has_frame_ && seq_ != my_seq);
    };

    if (!cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms), pred)) {
        return false; // timeout
    }
    if (stop_) return false;

    // Вариант 1 (быстрее): shallow copy. Подходит, если out используется сразу и вы не держите его "долго".
    out = last_;

    // Вариант 2 (безопаснее при асинхронной долгой обработке): out = last_.clone();
    // out = last_.clone();

    return true;
}

void FrameStore::stop() {
    {
        std::lock_guard<std::mutex> lk(m_);
        stop_ = true;
    }
    cv_.notify_all();
}

bool FrameStore::isStopped() const {
    std::lock_guard<std::mutex> lk(m_);
    return stop_;
}
