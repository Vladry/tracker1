#pragma once
#include <opencv2/opencv.hpp>
#include <condition_variable>
#include <mutex>

// FrameStore: хранит "последний кадр".
// Потребитель ждёт (sleep) на condition_variable.
// Производитель кладёт кадр и будит (notify_one).
class FrameStore {
public:
    FrameStore() = default;

    // Producer side (RTSP/appsink thread): кладёт новый кадр
    void setFrame(cv::Mat&& frame);

    // Optional: если вам удобнее пушить по const&
    void pushFrame(const cv::Mat& frame);

    // Consumer side (UI/tracker): ждёт новый кадр.
    // Возвращает true если кадр получен, false если stop().
    bool waitFrame(cv::Mat& out);

    // Останавливает ожидания, будит все ожидающие потоки.
    void stop();

    // Статус остановки (можно использовать в циклах)
    bool isStopped() const;

private:
    // Данные защищаем mutex'ом: cv::Mat - не тот тип, который безопасно менять без lock.
    cv::Mat last_; // - последний полученный кадр.
    bool has_frame_ = false; // - признак наличия первого кадра.

    // seq_ меняется только под mutex. Это "версия" кадра.
    // Потребитель ждёт, пока seq_ станет больше локально сохранённого.
    uint64_t seq_ = 0; // - счетчик версий для сигнализации новых кадров.

    mutable std::mutex m_; // - мьютекс для защиты состояния.
    std::condition_variable cv_; // - условная переменная для ожидания кадра.
    bool stop_ = false; // - флаг остановки ожиданий.
};
