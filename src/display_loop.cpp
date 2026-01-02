#include "display_loop.h"

#include <opencv2/opencv.hpp>
#include "other_handlers.h"
#include <atomic>
#include <chrono>
#include <iostream>

// Глобальные флаги (определены в main.cpp)
extern std::atomic<bool> g_running;
extern std::atomic<bool> g_rtsp_restart_requested;

// -------------------------------------------------
// Конструкторы (СИНХРОНИЗИРОВАНЫ с display_loop.h)
// -------------------------------------------------

// Дефолтная конфигурация
DisplayLoop::DisplayLoop(FrameStore& frames)
        : frames_(frames),
          cfg_(),
          limiter_(cfg_.target_fps)
{
}

// Явная конфигурация
DisplayLoop::DisplayLoop(FrameStore& frames, const Config& cfg)
        : frames_(frames),
          cfg_(cfg),
          limiter_(cfg_.target_fps)
{
}

// -------------------------------------------------
// UI loop
// -------------------------------------------------

void DisplayLoop::run()
{
    std::cout << "[DISPLAY] DisplayLoop started" << std::endl;

    cv::namedWindow(cfg_.window_name, cv::WINDOW_NORMAL);

    // восстановление выбора цели мышью (красный bbox)
    extern void on_mouse(int event, int x, int y, int flags, void* userdata);
    cv::setMouseCallback(cfg_.window_name, on_mouse, nullptr);


    cv::Mat frame;
    cv::Mat prime(64, 64, CV_8UC3, cv::Scalar(0, 0, 0));
    bool window_primed = false;

    while (g_running.load(std::memory_order_relaxed)) {

        // Ждём кадр из FrameStore (реальный API)
        const bool ok = frames_.waitFrame(frame);

        if (ok && !frame.empty()) {
            if (!window_primed) {
                window_primed = true;
                std::cout << "[DISPLAY] first frame received" << std::endl;
            }
            cv::imshow(cfg_.window_name, frame);
        } else if (!window_primed) {
            // Прокачиваем окно, чтобы клавиатура заработала
            cv::imshow(cfg_.window_name, prime);
            window_primed = true;
            std::cout << "[DISPLAY] window primed (no frames yet)" << std::endl;
        }

        int key_raw = cv::waitKey(1);
        int key = (key_raw >= 0) ? (key_raw & 0xFF) : -1;

        if (key == 27) { // ESC
            std::cout << "[KEY] ESC -> shutdown requested" << std::endl;
            g_running.store(false, std::memory_order_relaxed);
            break;
        }

        if (key == 'r' || key == 'R') {
            std::cout << "[KEY] R -> RTSP restart requested" << std::endl;
            g_rtsp_restart_requested.store(true, std::memory_order_relaxed);
        }

        // Ограничение FPS UI (реальный API)
        limiter_.tick();
    }

    cv::destroyWindow(cfg_.window_name);
    std::cout << "[DISPLAY] DisplayLoop finished" << std::endl;
}
