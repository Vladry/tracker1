#pragma once
#include <opencv2/opencv.hpp>
#include <atomic>
#include <string>
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
        int TARGET_FPS = 30; // - ограничение FPS для UI.
        std::string WINDOW_NAME = "video"; // - имя окна OpenCV.
    };

    // Базовый конструктор — дефолтная конфигурация
    explicit DisplayLoop(FrameStore& frames);


    // Расширенный конструктор — явная конфигурация
    DisplayLoop(FrameStore& frames, const Config& cfg);

    // Запускает UI-цикл и отображение кадров.
    void run();

private:
    FrameStore& frames_; // - источник кадров для отображения.
    Config cfg_; // - параметы окна и частоты обновления.
    RateLimiter limiter_; // - ограничитель частоты кадров.
};
