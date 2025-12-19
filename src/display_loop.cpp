#include "display_loop.h"

#include <atomic>
#include <iostream>

// Флаги из main.cpp (UI ставит, control-thread читает)
extern std::atomic<bool> g_running;
extern std::atomic<bool> g_rtsp_restart_requested;

void DisplayLoop::run() {
    cv::namedWindow(cfg_.window_name, cv::WINDOW_NORMAL);

    cv::Mat frame;
    while (g_running.load(std::memory_order_relaxed) && !stop_.load(std::memory_order_relaxed)) {
        const bool ok = frames_.waitFrame(frame, cfg_.wait_frame_ms);

        // Даже если кадра нет — UI поток не должен крутиться. waitFrame уже "усыпляет".
        if (ok && !frame.empty()) {
            cv::imshow(cfg_.window_name, frame);
        }

        // Без waitKey (не блокируемся здесь), но даём OpenCV обработать события.
        int key = cv::pollKey();
        if (key == 27) { // ESC
            std::cout << "[KEY] ESC -> выход из приложения" << std::endl;
            g_running.store(false, std::memory_order_relaxed);
            stop_.store(true, std::memory_order_relaxed);
        } else if (key == 'r' || key == 'R') {
            std::cout << "[KEY] RTSP restart requested" << std::endl;
            g_rtsp_restart_requested.store(true, std::memory_order_relaxed);
        }

        // Ограничиваем FPS — это часто главный фикс "86% CPU в main thread".
        limiter_.tick();
    }
}
