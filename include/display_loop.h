#pragma once
#include <opencv2/opencv.hpp>
#include <atomic>
#include "frame_store.h"
#include "rate_limiter.h"

// DisplayLoop: пример "правильного" UI-потока.
// 1) ждёт кадр через FrameStore::waitFrame (поток спит)
// 2) рисует и вызывает pollKey()
// 3) ограничивает FPS, чтобы не грузить CPU даже при частом wake-up
class DisplayLoop {
public:
    struct Config {
        int wait_frame_ms = 30;    // сколько ждать новый кадр
        int target_fps    = 30;    // ограничение FPS для UI
        std::string window_name = "video";
    };

    DisplayLoop(FrameStore& frames, std::atomic<bool>& stop_flag, Config cfg)
            : frames_(frames), stop_(stop_flag), cfg_(std::move(cfg)), limiter_(cfg_.target_fps) {}

    void run();

private:
    FrameStore& frames_;
    std::atomic<bool>& stop_;
    Config cfg_;
    RateLimiter limiter_;
};
