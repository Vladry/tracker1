#pragma once
#include <chrono>
#include <thread>

// RateLimiter: ограничивает частоту выполнения цикла (например, UI).
// Используйте tick() в конце итерации: он "усыпит" поток до следующего слота.
class RateLimiter {
public:
    explicit RateLimiter(int target_fps)
            : period_(target_fps > 0 ? std::chrono::microseconds(1000000 / target_fps)
                                     : std::chrono::microseconds(0)),
              next_(std::chrono::steady_clock::now()) {}

    // Дожидается следующего "слота" времени для ограничения FPS.
    void tick() {
        if (period_.count() <= 0) return;
        next_ += period_;
        std::this_thread::sleep_until(next_);
    }

    // Сбрасывает внутренний таймер в текущий момент.
    void reset() {
        next_ = std::chrono::steady_clock::now();
    }

private:
    std::chrono::microseconds period_; // - period_: длительность одного шага.
    std::chrono::steady_clock::time_point next_; // - next_: время следующего шага.
};
