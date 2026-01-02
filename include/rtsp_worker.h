#pragma once

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

#include "frame_store.h"
#include "config.h"

/*
 * RtspWorker
 * ==========
 * Назначение:
 *  - Поднять GStreamer pipeline для RTSP-камеры (обычно по UDP).
 *  - Доставлять кадры в FrameStore (последний кадр + wait/notify).
 *
 * Ключевая особенность (для стабильного "переподключения"):
 *  - При stop() пайплайн ВСЕГДА переводится в GST_STATE_NULL и уничтожается (unref).
 *  - При start() пайплайн создаётся заново "с нуля".
 *
 * Это критично для RTSP/UDP: иначе камера/сессия часто зависает в PAUSED→PLAYING (ASYNC).
 */
class RtspWorker {
public:
    // Состояние жизненного цикла RTSP-источника.
    enum class State : uint8_t {
        STOPPED = 0,   // пайплайна нет
        STARTING,      // создаём и запускаем пайплайн
        RUNNING,       // кадры идут
        STOPPING,      // teardown / переход в NULL
        ERROR          // ошибка/зависание (рабочий поток завершился с ошибкой)
    };

public:
    RtspWorker(FrameStore& store, toml::table& tbl);
    ~RtspWorker();

    // Запуск RTSP-потока (если уже запущен — ничего не делает).
    void start();

    // Полная остановка RTSP-потока (teardown + destroy).
    void stop();

    // Жёсткий рестарт: stop(); пауза; start();
    void restart();

    // Текущее состояние воркера.
    State state() const { return state_.load(std::memory_order_relaxed); }

    // Был ли получен хотя бы один кадр в текущей сессии.
    bool gotFirstSample() const { return gotFirstSample_.load(std::memory_order_relaxed); }

private:
    struct RtspConfig {
        // RTSP URL камеры. Пример:
        // "rtsp://192.168.144.25:8554/main.264" - для камеры SIYI
        std::string url = "rtsp://192.168.144.25:8554/main.264";

        // Протоколы rtspsrc:
        //(битовая маска) 1 = UDP, 4 = TCP
        int protocols = 1;

        // rtspsrc::latency (мс). 0 = минимальная задержка (но больше риск нестабильности)
        int latency_ms = 0;

        // rtspsrc::timeout / tcp-timeout (микросекунды).
        // Обычно достаточно 2с, чтобы не висеть бесконечно при старте.
        uint64_t timeout_us = 2'000'000;
        uint64_t tcp_timeout_us = 2'000'000;

        // Принудительные caps после декодера (capsfilter).
        // На RK (mppvideodec) типичный стабильный формат для OpenCV: NV12.
        std::string caps_force = "video/x-raw,format=NV12";

        // Пауза после stop() перед повторным стартом (мс).
        // Нужна, чтобы камера/стек RTSP успели закрыть сессию и освободить UDP порты.
        int restart_delay_ms = 300;

        bool logger_on = false;

    };


    // Главный поток RTSP-обработчика.
    void threadMain();

    // Создать новый пайплайн. Возвращает true при успехе.
    bool buildPipeline();

    // Безопасно уничтожить текущий пайплайн (внутри рабочего потока).
    void teardownPipeline();

    // Сигнал "разбудить" loop, чтобы быстрее отреагировать на stop().
    void pokeBus();

    // Callback appsink: пришёл новый sample.
    static GstFlowReturn onNewSample(GstAppSink* sink, gpointer user_data);

    // Callback rtspsrc: добавился pad (подцепляем к depay).
    static void onPadAdded(GstElement* src, GstPad* new_pad, gpointer user_data);

private:
    FrameStore& store_;
    RtspConfig cfg_;

    std::atomic<State> state_{State::STOPPED};
    std::atomic<bool> running_{false};         // "рабочий поток должен работать"
    std::atomic<bool> stopRequested_{false};   // "нужно остановиться"
    std::atomic<bool> gotFirstSample_{false};

    std::thread th_;

    // GStreamer объекты (используются только в worker thread, кроме pokeBus()).
    GstElement* pipeline_{nullptr};
    GstElement* src_{nullptr};
    GstElement* depay_{nullptr};
    GstElement* parse_{nullptr};
    GstElement* dec_{nullptr};
    GstElement* force_caps_{nullptr};
    GstElement* sink_{nullptr};
    GstBus* bus_{nullptr};

    // Мьютекс только для безопасного доступа к bus_ в pokeBus().
    mutable std::mutex bus_mu_;
    bool load_rtsp_config(toml::table& tbl);
};
