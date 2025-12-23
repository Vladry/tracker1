#include "rtsp_worker.h"
#include "config.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstring>
#include <iostream>

using namespace std::chrono;

RtspWorker::RtspWorker(FrameStore& store, const RtspConfig& cfg)
    : store_(store), cfg_(cfg) {}

RtspWorker::~RtspWorker() {
    stop();
}

void RtspWorker::start() {
    // Уже запущено — ничего не делаем.
    if (running_.load(std::memory_order_acquire)) return;

    // Сбрасываем флаги на новую "сессию".
    gotFirstSample_.store(false, std::memory_order_release);
    stopRequested_.store(false, std::memory_order_release);

    running_.store(true, std::memory_order_release);
    state_.store(State::STARTING, std::memory_order_release);

    th_ = std::thread(&RtspWorker::threadMain, this);
}

void RtspWorker::stop() {
    // Если не было запуска — нечего останавливать.
    if (!running_.exchange(false, std::memory_order_acq_rel)) return;

    state_.store(State::STOPPING, std::memory_order_release);
    stopRequested_.store(true, std::memory_order_release);

    // Разбудим loop, если он ждёт сообщение на bus.
    pokeBus();

    if (th_.joinable()) th_.join();

    // Рабочий поток гарантированно почистил pipeline_ и bus_.
    state_.store(State::STOPPED, std::memory_order_release);
}

void RtspWorker::restart() {
    stop();

    // Пауза нужна, чтобы камера и сетевой стек успели закрыть сессию/порты.
    if (cfg_.restart_delay_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(cfg_.restart_delay_ms));
    }

    start();
}

void RtspWorker::pokeBus() {
    std::lock_guard<std::mutex> lk(bus_mu_);
    if (!bus_ || !pipeline_) return;

    // Сообщение "для себя", чтобы gst_bus_timed_pop() проснулся раньше таймаута.
    GstStructure* st = gst_structure_new_empty("app-stop");
    GstMessage* msg = gst_message_new_application(GST_OBJECT(pipeline_), st);
    gst_bus_post(bus_, msg);
}

void RtspWorker::threadMain() {
    if (cfg_.verbose) {
        std::cerr << "RTSP: worker thread started" << std::endl;
        std::cerr.flush();
    }

    if (!buildPipeline()) {
        state_.store(State::ERROR, std::memory_order_release);
        running_.store(false, std::memory_order_release);
        if (cfg_.verbose) {
            std::cerr << "RTSP: buildPipeline() failed" << std::endl;
            std::cerr.flush();
        }
        teardownPipeline();
        return;
    }

    // Запускаем пайплайн.
    gst_element_set_state(pipeline_, GST_STATE_PLAYING);

    // Ждём перехода в PLAYING ограниченное время.
    {
        GstState cur = GST_STATE_NULL, pending = GST_STATE_NULL;
        GstStateChangeReturn ret = gst_element_get_state(
            pipeline_, &cur, &pending,
            (GstClockTime)cfg_.start_timeout_ms * GST_MSECOND
        );

        if (ret == GST_STATE_CHANGE_FAILURE) {
            if (cfg_.verbose) {
                std::cerr << "RTSP: state change FAILURE on start" << std::endl;
                std::cerr.flush();
            }
            state_.store(State::ERROR, std::memory_order_release);
            teardownPipeline();
            running_.store(false, std::memory_order_release);
            return;
        }
        // Даже если ASYNC, мы всё равно продолжаем и слушаем bus — но стартовый таймаут уже отработал.
    }

    state_.store(State::RUNNING, std::memory_order_release);

    // Основной цикл: ждём сообщения на bus, периодически проверяем stopRequested_.
    while (running_.load(std::memory_order_acquire) && !stopRequested_.load(std::memory_order_acquire)) {
        GstMessage* msg = gst_bus_timed_pop(bus_, 200 * GST_MSECOND);
        if (!msg) continue;

        switch (GST_MESSAGE_TYPE(msg)) {
            case GST_MESSAGE_ERROR: {
                GError* err = nullptr;
                gchar* dbg = nullptr;
                gst_message_parse_error(msg, &err, &dbg);

                if (cfg_.verbose) {
                    std::cerr << "RTSP: GST_MESSAGE_ERROR: "
                              << (err ? err->message : "(null)") << std::endl;
                    if (dbg) std::cerr << "RTSP: debug: " << dbg << std::endl;
                    std::cerr.flush();
                }

                if (err) g_error_free(err);
                if (dbg) g_free(dbg);

                state_.store(State::ERROR, std::memory_order_release);
                stopRequested_.store(true, std::memory_order_release);
                break;
            }
            case GST_MESSAGE_EOS:
                if (cfg_.verbose) {
                    std::cerr << "RTSP: EOS" << std::endl;
                    std::cerr.flush();
                }
                stopRequested_.store(true, std::memory_order_release);
                break;

            default:
                // APPLICATION и прочие — игнорируем.
                break;
        }

        gst_message_unref(msg);
    }

    // Жёсткая остановка: перевод в NULL + уничтожение объектов.
    teardownPipeline();

    if (cfg_.verbose) {
        std::cerr << "RTSP: worker thread stopped" << std::endl;
        std::cerr.flush();
    }
}

bool RtspWorker::buildPipeline() {
    teardownPipeline(); // на всякий случай

    // Создаём элементы.
    // rtspsrc даёт динамический src pad -> подключаем в onPadAdded().
    pipeline_   = gst_pipeline_new("rtsp-pipeline");
    src_        = gst_element_factory_make("rtspsrc", "src");
    depay_      = gst_element_factory_make("rtph265depay", "depay");
    parse_      = gst_element_factory_make("h265parse", "parse");
    dec_        = gst_element_factory_make("mppvideodec", "dec");
    force_caps_ = gst_element_factory_make("capsfilter", "force_caps");
    sink_       = gst_element_factory_make("appsink", "sink");

    if (!pipeline_ || !src_ || !depay_ || !parse_ || !dec_ || !force_caps_ || !sink_) {
        if (cfg_.verbose) {
            std::cerr << "RTSP: failed to create one or more GStreamer elements" << std::endl;
            std::cerr.flush();
        }
        return false;
    }

    // Настраиваем rtspsrc.
    g_object_set(G_OBJECT(src_), "location", cfg_.url.c_str(), nullptr);
    g_object_set(G_OBJECT(src_), "protocols", cfg_.protocols, nullptr);
    g_object_set(G_OBJECT(src_), "latency", cfg_.latency_ms, nullptr);
    g_object_set(G_OBJECT(src_), "timeout", (guint64)cfg_.timeout_us, nullptr);
    g_object_set(G_OBJECT(src_), "tcp-timeout", (guint64)cfg_.tcp_timeout_us, nullptr);

    // capsfilter после декодера.
    GstCaps* caps = gst_caps_from_string(cfg_.caps_force.c_str());
    g_object_set(G_OBJECT(force_caps_), "caps", caps, nullptr);
    gst_caps_unref(caps);

    // appsink — минимальная задержка, отдаём только последний кадр.
    g_object_set(G_OBJECT(sink_), "emit-signals", TRUE, nullptr);
    g_object_set(G_OBJECT(sink_), "sync", FALSE, nullptr);
    g_object_set(G_OBJECT(sink_), "max-buffers", 1, nullptr);
    g_object_set(G_OBJECT(sink_), "drop", TRUE, nullptr);

    // Callback на новые кадры.
    GstAppSinkCallbacks cbs{};
    cbs.new_sample = &RtspWorker::onNewSample;
    gst_app_sink_set_callbacks(GST_APP_SINK(sink_), &cbs, this, nullptr);

    // rtspsrc pad-added -> подключение к depay.
    g_signal_connect(src_, "pad-added", G_CALLBACK(&RtspWorker::onPadAdded), this);

    // Добавляем и линкуем статическую часть.
    gst_bin_add_many(GST_BIN(pipeline_), src_, depay_, parse_, dec_, force_caps_, sink_, nullptr);

    if (!gst_element_link_many(depay_, parse_, dec_, force_caps_, sink_, nullptr)) {
        if (cfg_.verbose) {
            std::cerr << "RTSP: failed to link depay->parse->dec->caps->sink" << std::endl;
            std::cerr.flush();
        }
        return false;
    }

    // Bus для обработки ошибок.
    bus_ = gst_element_get_bus(pipeline_);
    return true;
}

void RtspWorker::teardownPipeline() {
    // Важно: teardown делаем только один раз.
    if (!pipeline_) {
        std::lock_guard<std::mutex> lk(bus_mu_);
        if (bus_) {
            gst_object_unref(bus_);
            bus_ = nullptr;
        }
        return;
    }

    // 1) Переводим пайплайн в NULL — это и есть "teardown" RTSP-сессии.
    gst_element_set_state(pipeline_, GST_STATE_NULL);

    // 2) Ждём завершения перехода ограниченное время.
    GstState cur = GST_STATE_NULL, pending = GST_STATE_NULL;
    gst_element_get_state(
        pipeline_, &cur, &pending,
        (GstClockTime)cfg_.stop_timeout_ms * GST_MSECOND
    );

    // 3) Освобождаем bus (под мьютексом, т.к. stop() может дернуть pokeBus()).
    {
        std::lock_guard<std::mutex> lk(bus_mu_);
        if (bus_) {
            gst_object_unref(bus_);
            bus_ = nullptr;
        }
    }

    // 4) Освобождаем пайплайн и всё дерево элементов (unref pipeline достаточно).
    gst_object_unref(pipeline_);
    pipeline_ = nullptr;

    // Элементы обнулим для ясности.
    src_ = depay_ = parse_ = dec_ = force_caps_ = sink_ = nullptr;
}

GstFlowReturn RtspWorker::onNewSample(GstAppSink* sink, gpointer user_data) {
    auto* self = static_cast<RtspWorker*>(user_data);
    if (!self) return GST_FLOW_ERROR;

    GstSample* sample = gst_app_sink_pull_sample(sink);
    if (!sample) return GST_FLOW_ERROR;

    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstCaps* caps = gst_sample_get_caps(sample);

    if (!buffer || !caps) {
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }

    // Получаем размеры из caps (ожидаем NV12).
    GstStructure* s = gst_caps_get_structure(caps, 0);
    int width = 0, height = 0;
    gst_structure_get_int(s, "width", &width);
    gst_structure_get_int(s, "height", &height);

    GstMapInfo map{};
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }

    // NV12: Y plane (H*W) + interleaved UV (H/2*W) => всего H*W*3/2 байт.
    // OpenCV удобно конвертировать через COLOR_YUV2BGR_NV12.
    cv::Mat yuv(height + height / 2, width, CV_8UC1, (void*)map.data);
    cv::Mat bgr;
    cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12);

    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);

    // Отмечаем первый кадр.
    if (!self->gotFirstSample_.exchange(true, std::memory_order_acq_rel)) {
        if (self->cfg_.verbose) {
            std::cerr << "RTSP: first sample received" << std::endl;
            std::cerr.flush();
        }
    }

    // Кладём кадр в FrameStore (move, чтобы не копировать лишний раз).
    self->store_.setFrame(std::move(bgr));

    return GST_FLOW_OK;
}

void RtspWorker::onPadAdded(GstElement* src, GstPad* new_pad, gpointer user_data) {
    auto* self = static_cast<RtspWorker*>(user_data);
    if (!self || !self->depay_) return;

    GstCaps* caps = gst_pad_get_current_caps(new_pad);
    if (!caps) caps = gst_pad_query_caps(new_pad, nullptr);
    if (!caps) return;

    // Важно: фильтруем только RTP video.
    GstStructure* str = gst_caps_get_structure(caps, 0);
    const char* name = gst_structure_get_name(str);

    // Ожидаем "application/x-rtp".
    if (!name || std::strcmp(name, "application/x-rtp") != 0) {
        gst_caps_unref(caps);
        return;
    }

    GstPad* sinkpad = gst_element_get_static_pad(self->depay_, "sink");
    if (!sinkpad) {
        gst_caps_unref(caps);
        return;
    }

    if (!gst_pad_is_linked(sinkpad)) {
        GstPadLinkReturn ret = gst_pad_link(new_pad, sinkpad);
        if (self->cfg_.verbose) {
            std::cerr << "RTSP: [pad-added] link result = " << ret << std::endl;
            std::cerr.flush();
        }
    }

    gst_object_unref(sinkpad);
    gst_caps_unref(caps);
}
