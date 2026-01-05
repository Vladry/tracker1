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
    // Инициализирует RTSP-воркер и считывает конфигурацию.
    RtspWorker(FrameStore& store, toml::table& tbl);
    // Останавливает поток и освобождает ресурсы.
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
        std::string URL = "rtsp://192.168.144.25:8554/main.264";

        // Протоколы rtspsrc:
        //(битовая маска) 1 = UDP, 4 = TCP
        int PROTOCOLS = 1;

        // rtspsrc::latency (мс). 0 = минимальная задержка (но больше риск нестабильности)
        int LATENCY_MS = 0;

        // rtspsrc::timeout / tcp-timeout (микросекунды).
        // Обычно достаточно 2с, чтобы не висеть бесконечно при старте.
        uint64_t TIMEOUT_US = 2'000'000;
        uint64_t TCP_TIMEOUT_US = 2'000'000;

        // Принудительные caps после декодера (capsfilter).
        // На RK (mppvideodec) типичный стабильный формат для OpenCV: NV12.
        std::string CAPS_FORCE = "video/x-raw,format=NV12";

        // Пауза после stop() перед повторным стартом (мс).
        // Нужна, чтобы камера/стек RTSP успели закрыть сессию и освободить UDP порты.
        int RESTART_DELAY_MS = 300;

        bool LOGGER_ON = false; // - LOGGER_ON: включение подробного логирования RTSP.

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

    // Callback rtspsrc: добаился pad (подцепляем к depay).
    static void onPadAdded(GstElement* src, GstPad* new_pad, gpointer user_data);

private:
    FrameStore& store_; // - store_: приёмник кадров для передачи в UI/трекер.
    RtspConfig cfg_; // - cfg_: параметры RTSP-пайплайна.

    std::atomic<State> state_{State::STOPPED}; // - state_: текущее состояние жизненного цикла.
    std::atomic<bool> running_{false}; // - running_: флаг работы потока.
    std::atomic<bool> stopRequested_{false}; // - stopRequested_: запрос на остановку.
    std::atomic<bool> gotFirstSample_{false}; // - gotFirstSample_: признак первого кадра.

    std::thread th_; // - th_: поток с RTSP-циклом.

    // GStreamer объекты (используются только в worker thread, кроме pokeBus()).
    GstElement* pipeline_{nullptr}; // - pipeline_: корневой пайплайн GStreamer.
    GstElement* src_{nullptr}; // - src_: элемент rtspsrc.
    GstElement* depay_{nullptr}; // - depay_: depayloader RTP.
    GstElement* parse_{nullptr}; // - parse_: парсер H.265.
    GstElement* dec_{nullptr}; // - dec_: декодер видео.
    GstElement* force_caps_{nullptr}; // - force_caps_: capsfilter после декодера.
    GstElement* sink_{nullptr}; // - sink_: appsink для получения кадров.
    GstBus* bus_{nullptr}; // - bus_: шина сообщений GStreamer.

    // Мьютекс только для безопасного доступа к bus_ в pokeBus().
    mutable std::mutex bus_mu_; // - bus_mu_: защита доступа к bus_ в pokeBus().
    // Загружает конфигурацию RTSP из TOML.
    bool load_rtsp_config(toml::table& tbl);
};
