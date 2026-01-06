#pragma once
#include <condition_variable>
#include <deque>
#include <mutex>

// BlockingQueue<T>: классическая очередь для worker-потоков.
// pop() блокируется, пока нет элементов или пока не вызван stop().
template<typename T>
class BlockingQueue {
public:
    BlockingQueue() = default;

    // Помещает элемент в очередь, перемещая значение.
    void push(T&& v) {
        {
            std::lock_guard<std::mutex> lk(m_);
            if (stop_) return;
            q_.emplace_back(std::move(v));
        }
        cv_.notify_one();
    }

    // Помещает элемент в очередь копированием.
    void push(const T& v) {
        {
            std::lock_guard<std::mutex> lk(m_);
            if (stop_) return;
            q_.push_back(v);
        }
        cv_.notify_one();
    }

    // Блокирует поток, пока нет элемента или stop().
    // Возвращает false, если очередь остановлена и пуста.
    bool pop(T& out) {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&]{ return stop_ || !q_.empty(); });

        if (stop_ && q_.empty()) return false;

        out = std::move(q_.front());
        q_.pop_front();
        return true;
    }

    // Неблокирующее извлечение (опционально)
    bool try_pop(T& out) {
        std::lock_guard<std::mutex> lk(m_);
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        return true;
    }

    // Переводит очередь в состояние остановки и будит ожидающие потоки.
    void stop() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
        }
        cv_.notify_all();
    }

    // Сообщает, остановлена ли очередь.
    bool stopped() const {
        std::lock_guard<std::mutex> lk(m_);
        return stop_;
    }

    // Возвращает текущее количество элементов в очереди.
    size_t size() const {
        std::lock_guard<std::mutex> lk(m_);
        return q_.size();
    }

private:
    mutable std::mutex m_; // - мьютекс для защиты очереди.
    std::condition_variable cv_; // - условная переменная для ожидания данных.
    std::deque<T> q_; // - контейнер для хранения элементов.
    bool stop_ = false; // - флаг остановки очереди.
};
