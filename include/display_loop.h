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
// 2) рисует и ызывает pollKey()
// 3) ограничивает FPS, чтобы не грузить CPU даже при частом wake-up
class DisplayLoop {
public:
    struct Config {
        int target_fps = 30; // - target_fps: ограничение FPS для UI.
        std::string window_name = "video"; // - window_name: имя окна OpenCV.
    };

    // Базовый конструктор — дефолтная конфигурация
    explicit DisplayLoop(FrameStore& frames);


    // Расширенный конструктор — явная конфигурация
    DisplayLoop(FrameStore& frames, const Config& cfg);

    // Запускает UI-цикл и отображение кадров.
    void run();

private:
    FrameStore& frames_; // - frames_: источник кадров для отображения.
    Config cfg_; // - cfg_: параметры окна и частоты обновления.
    RateLimiter limiter_; // - limiter_: ограничитель частоты кадров.
};
