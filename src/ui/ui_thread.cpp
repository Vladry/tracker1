#include "ui/ui_thread.h"

#include <chrono>

namespace ui {

    UiThread::UiThread(std::string window_name, int target_fps)
            : window_(std::move(window_name)), target_fps_(target_fps) {
    }

    UiThread::~UiThread() {
        stop();
    }

    void UiThread::start() {
        if (running_.exchange(true)) return;

        th_ = std::thread([this] { run(); });
    }

    void UiThread::stop() {
        if (!running_.exchange(false)) return;
        cv_.notify_all();
        if (th_.joinable()) th_.join();
    }

    void UiThread::submitFrame(const cv::Mat& frame) {
        {
            std::lock_guard<std::mutex> lk(m_);
            // clone, чтобы processing thread мог переиспользовать/освободить свой Mat
            last_frame_ = frame.clone();
            has_frame_ = true;
        }
        cv_.notify_one();
    }

    void UiThread::onMouse(int event, int x, int y, int, void* userdata) {
        if (event != cv::EVENT_LBUTTONDOWN) return;
        auto* self = static_cast<UiThread*>(userdata);
        if (!self) return;

        self->click_.x.store(x, std::memory_order_release);
        self->click_.y.store(y, std::memory_order_release);
        self->click_.seq.fetch_add(1, std::memory_order_acq_rel);
    }

    void UiThread::run() {
        // Window belongs to UI thread
        cv::namedWindow(window_, cv::WINDOW_NORMAL);
        cv::setMouseCallback(window_, &UiThread::onMouse, this);

        using clock = std::chrono::steady_clock;
        const auto frame_period = std::chrono::milliseconds(
                target_fps_ > 0 ? (1000 / target_fps_) : 33
        );

        cv::Mat local;

        auto next_tick = clock::now();

        while (running_.load(std::memory_order_acquire)) {

            // Ждём либо новый кадр, либо следующий UI-tick, либо stop
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait_until(lk, next_tick, [&] {
                    return !running_.load(std::memory_order_acquire) || has_frame_;
                });

                if (!running_.load(std::memory_order_acquire))
                    break;

                if (has_frame_) {
                    local = last_frame_;
                    has_frame_ = false;
                }
            }

            // Даже если нового кадра не было, мы “дёргаем” HighGUI события по таймеру.
            // Это критично для стабильной отрисовки и mouse событий без waitKey.
            if (!local.empty()) {
                cv::imshow(window_, local);
            }
            cv::pollKey();

            next_tick += frame_period;

            // если мы сильно отстали, пересинхронизируемся
            auto now = clock::now();
            if (now > next_tick + frame_period * 3) {
                next_tick = now + frame_period;
            }
        }

        // При выходе можно оставить окно; OpenCV обычно сам корректно завершает.
    }

} // namespace ui
