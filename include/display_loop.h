#pragma once
#include <opencv2/opencv.hpp>
#include <atomic>
#include "frame_store.h"
#include "rate_limiter.h"


// Глобальные флаги (определены в main.cpp)
extern std::atomic<bool> g_running;
extern std::atomic<bool> g_rtsp_restart_requested;


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

    // Базовый конструктор — дефолтная конфигурация
    explicit DisplayLoop(FrameStore& frames);


    // Расширенный конструктор — явная конфигурация
    DisplayLoop(FrameStore& frames, const Config& cfg);

    void run();

private:
    FrameStore& frames_;
    Config cfg_;
    RateLimiter limiter_;
};
