#include "display_loop.h"

void DisplayLoop::run() {
    cv::namedWindow(cfg_.window_name, cv::WINDOW_NORMAL);

    cv::Mat frame;
    while (!stop_.load(std::memory_order_relaxed)) {
        const bool ok = frames_.waitFrame(frame, cfg_.wait_frame_ms);

        // Даже если кадра нет — UI поток не должен крутиться. waitFrame уже "усыпляет".
        if (ok && !frame.empty()) {
            cv::imshow(cfg_.window_name, frame);
        }

        // Без waitKey (не блокируемся здесь), но даём OpenCV обработать события.
        cv::pollKey();

        // Ограничиваем FPS — это часто главный фикс "86% CPU в main thread".
        limiter_.tick();
    }
}
