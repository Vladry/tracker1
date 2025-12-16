#pragma once
#include <opencv2/opencv.hpp>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <string>

namespace ui {

// Поток UI: показывает последний кадр с заданной частотой.
// Без waitKey. Для HighGUI events используется pollKey().
// Клики мыши складываются в атомик и забираются обработчиком в processing-потоке.
    class UiThread {
    public:
        struct Click {
            std::atomic<int> x{-1};
            std::atomic<int> y{-1};
            std::atomic<int> seq{0};   // увеличивается при каждом клике
        };

        UiThread(std::string window_name, int target_fps = 30);
        ~UiThread();

        UiThread(const UiThread&) = delete;
        UiThread& operator=(const UiThread&) = delete;

        void start();
        void stop();

        // Отдать кадр на отображение (копируется/клонируется внутри, чтобы не зависеть от жизненного цикла frame).
        void submitFrame(const cv::Mat& frame);

        // Доступ к кликам мыши (UI thread пишет, processing thread читает).
        Click& clickState() { return click_; }

    private:
        static void onMouse(int event, int x, int y, int flags, void* userdata);

        void run();

    private:
        std::string window_;
        int target_fps_;

        std::thread th_;
        std::atomic<bool> running_{false};

        std::mutex m_;
        std::condition_variable cv_;
        cv::Mat last_frame_;
        bool has_frame_ = false;

        Click click_;
    };

} // namespace ui
