#include "rtsp_worker.h"
#include "config.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstring>
#include <iostream>
#include <utility>

using namespace std::chrono;

RtspWorker::RtspWorker(FrameStore& store, toml::table& tbl)
        : store_(store) {
    load_rtsp_config(tbl);
}

RtspWorker::~RtspWorker() {
    stop();
}

void RtspWorker::start() {
    if (running_.load(std::memory_order_acquire)) return;

    gotFirstSample_.store(false, std::memory_order_release);
    stopRequested_.store(false, std::memory_order_release);

    running_.store(true, std::memory_order_release);
    state_.store(State::STARTING, std::memory_order_release);

    th_ = std::thread(&RtspWorker::threadMain, this);
}

void RtspWorker::stop() {
    if (!running_.exchange(false, std::memory_order_acq_rel)) return;

    state_.store(State::STOPPING, std::memory_order_release);
    stopRequested_.store(true, std::memory_order_release);

    pokeBus();

    if (th_.joinable()) th_.join();

    state_.store(State::STOPPED, std::memory_order_release);
}

void RtspWorker::restart() {
    stop();

    if (cfg_.RESTART_DELAY_MS > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(cfg_.RESTART_DELAY_MS));
    }

    start();
}

void RtspWorker::pokeBus() {
    std::lock_guard<std::mutex> lk(bus_mu_);
    if (!bus_ || !pipeline_) return;

    GstStructure* st = gst_structure_new_empty("app-stop");
    GstMessage* msg = gst_message_new_application(GST_OBJECT(pipeline_), st);
    gst_bus_post(bus_, msg);
}

void RtspWorker::threadMain() {
    if (cfg_.LOGGER_ON) {
        std::cerr << "RTSP: worker thread started" << std::endl;
        std::cerr.flush();
    }

    if (!buildPipeline()) {
        state_.store(State::ERROR, std::memory_order_release);
        running_.store(false, std::memory_order_release);
        if (cfg_.LOGGER_ON) {
            std::cerr << "RTSP: buildPipeline() failed" << std::endl;
            std::cerr.flush();
        }
        teardownPipeline();
        return;
    }

    GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        if (cfg_.LOGGER_ON) {
            std::cerr << "RTSP: state change FAILURE on start" << std::endl;
            std::cerr.flush();
        }
        state_.store(State::ERROR, std::memory_order_release);
        teardownPipeline();
        running_.store(false, std::memory_order_release);
        return;
    }

    state_.store(State::RUNNING, std::memory_order_release);

    while (running_.load(std::memory_order_acquire) &&
           !stopRequested_.load(std::memory_order_acquire)) {

        GstMessage* msg = gst_bus_timed_pop(bus_, 200 * GST_MSECOND);
        if (!msg) continue;

        switch (GST_MESSAGE_TYPE(msg)) {
            case GST_MESSAGE_ERROR: {
                GError* err = nullptr;
                gchar* dbg = nullptr;
                gst_message_parse_error(msg, &err, &dbg);

                if (cfg_.LOGGER_ON) {
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
                if (cfg_.LOGGER_ON) {
                    std::cerr << "RTSP: EOS" << std::endl;
                    std::cerr.flush();
                }
                stopRequested_.store(true, std::memory_order_release);
                break;

            default:
                break;
        }

        gst_message_unref(msg);
    }

    teardownPipeline();

    if (cfg_.LOGGER_ON) {
        std::cerr << "RTSP: worker thread stopped" << std::endl;
        std::cerr.flush();
    }
}

bool RtspWorker::buildPipeline() {
    teardownPipeline();

    pipeline_   = gst_pipeline_new("rtsp-pipeline");
    src_        = gst_element_factory_make("rtspsrc", "src");
    depay_      = gst_element_factory_make("rtph265depay", "depay");
    parse_      = gst_element_factory_make("h265parse", "parse");
    dec_        = gst_element_factory_make("mppvideodec", "dec");
    force_caps_ = gst_element_factory_make("capsfilter", "force_caps");
    sink_       = gst_element_factory_make("appsink", "sink");

    if (!pipeline_ || !src_ || !depay_ || !parse_ || !dec_ || !force_caps_ || !sink_) {
        if (cfg_.LOGGER_ON) {
            std::cerr << "RTSP: failed to create one or more GStreamer elements" << std::endl;
            std::cerr.flush();
        }
        return false;
    }

    g_object_set(G_OBJECT(src_), "location", cfg_.URL.c_str(), nullptr);
    g_object_set(G_OBJECT(src_), "protocols", cfg_.PROTOCOLS, nullptr);
    g_object_set(G_OBJECT(src_), "latency", cfg_.LATENCY_MS, nullptr);
    g_object_set(G_OBJECT(src_), "timeout", (guint64)cfg_.TIMEOUT_US, nullptr);
    g_object_set(G_OBJECT(src_), "tcp-timeout", (guint64)cfg_.TCP_TIMEOUT_US, nullptr);

    // capsfilter после декодера
    GstCaps* caps = gst_caps_from_string(cfg_.CAPS_FORCE.c_str());
    g_object_set(G_OBJECT(force_caps_), "caps", caps, nullptr);
    gst_caps_unref(caps);

    g_object_set(G_OBJECT(sink_), "emit-signals", TRUE, nullptr);
    g_object_set(G_OBJECT(sink_), "sync", FALSE, nullptr);
    g_object_set(G_OBJECT(sink_), "max-buffers", 1, nullptr);
    g_object_set(G_OBJECT(sink_), "drop", TRUE, nullptr);

    GstAppSinkCallbacks cbs{};
    cbs.new_sample = &RtspWorker::onNewSample;
    gst_app_sink_set_callbacks(GST_APP_SINK(sink_), &cbs, this, nullptr);

    g_signal_connect(src_, "pad-added", G_CALLBACK(&RtspWorker::onPadAdded), this);

    gst_bin_add_many(GST_BIN(pipeline_), src_, depay_, parse_, dec_, force_caps_, sink_, nullptr);

    if (!gst_element_link_many(depay_, parse_, dec_, force_caps_, sink_, nullptr)) {
        if (cfg_.LOGGER_ON) {
            std::cerr << "RTSP: failed to link depay->parse->dec->caps->sink" << std::endl;
            std::cerr.flush();
        }
        return false;
    }

    bus_ = gst_element_get_bus(pipeline_);
    return true;
}

void RtspWorker::teardownPipeline() {
    if (!pipeline_) {
        std::lock_guard<std::mutex> lk(bus_mu_);
        if (bus_) {
            gst_object_unref(bus_);
            bus_ = nullptr;
        }
        return;
    }

    gst_element_set_state(pipeline_, GST_STATE_NULL);

    {
        std::lock_guard<std::mutex> lk(bus_mu_);
        if (bus_) {
            gst_object_unref(bus_);
            bus_ = nullptr;
        }
    }

    gst_object_unref(pipeline_);
    pipeline_ = nullptr;

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

    GstStructure* s = gst_caps_get_structure(caps, 0);
    int width = 0, height = 0;
    gst_structure_get_int(s, "width", &width);
    gst_structure_get_int(s, "height", &height);

    GstMapInfo map{};
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }

    cv::Mat yuv(height + height / 2, width, CV_8UC1, (void*)map.data);
    cv::Mat bgr;
    cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12);

    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);

    if (!self->gotFirstSample_.exchange(true, std::memory_order_acq_rel)) {
        if (self->cfg_.LOGGER_ON) {
            std::cerr << "RTSP: first sample received" << std::endl;
            std::cerr.flush();
        }
    }

    self->store_.setFrame(std::move(bgr));
    return GST_FLOW_OK;
}

void RtspWorker::onPadAdded(GstElement* src, GstPad* new_pad, gpointer user_data) {
    auto* self = static_cast<RtspWorker*>(user_data);
    if (!self || !self->depay_) return;

    GstCaps* caps = gst_pad_get_current_caps(new_pad);
    if (!caps) caps = gst_pad_query_caps(new_pad, nullptr);
    if (!caps) return;

    GstStructure* str = gst_caps_get_structure(caps, 0);
    const char* name = gst_structure_get_name(str);

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
        if (self->cfg_.LOGGER_ON) {
            std::cerr << "RTSP: [pad-added] link result = " << ret << std::endl;
            std::cerr.flush();
        }
    }

    gst_object_unref(sinkpad);
    gst_caps_unref(caps);
}

bool RtspWorker::load_rtsp_config(toml::table &tbl) {
    // ----------------------------- [rtsp] -----------------------------
    try {
        const auto *rtsp_node = tbl.get("rtsp");
        if (!rtsp_node) {
            throw std::runtime_error("missing [rtsp] table");
        }
        const auto *rtsp = rtsp_node->as_table();
        if (!rtsp) {
            throw std::runtime_error("invalid [rtsp] table");
        }
        cfg_.URL = read_required<std::string>(*rtsp, "URL");
        cfg_.PROTOCOLS = read_required<int>(*rtsp, "PROTOCOLS");
        cfg_.LATENCY_MS = read_required<int>(*rtsp, "LATENCY_MS");
        cfg_.TIMEOUT_US = read_required<std::uint64_t>(*rtsp, "TIMEOUT_US");
        cfg_.TCP_TIMEOUT_US = read_required<std::uint64_t>(*rtsp, "TCP_TIMEOUT_US");
        cfg_.CAPS_FORCE = read_required<std::string>(*rtsp, "CAPS_FORCE");
        cfg_.RESTART_DELAY_MS = read_required<int>(*rtsp, "RESTART_DELAY_MS");
        LoggingConfig log_cfg{};
        load_logging_config(tbl, log_cfg);
        cfg_.LOGGER_ON = log_cfg.RTSP_LEVEL_LOGGER_ON;


        return true;

    } catch (const std::exception &e) {
        std::cerr << "rtsp config load failed  " << e.what() << std::endl;
        return false;
    }
};
