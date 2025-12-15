#include "rtsp_worker.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

RtspWorker::RtspWorker(FrameStore& store, const Config& cfg)
: store_(store), cfg_(cfg) {}

RtspWorker::~RtspWorker() {
    stop();
}

void RtspWorker::start() {
    if (running_.exchange(true)) return;
    th_ = std::thread(&RtspWorker::threadMain, this);
}

void RtspWorker::stop() {
    if (!running_.exchange(false)) return;
    if (th_.joinable()) th_.join();
}

void RtspWorker::threadMain() {
    if (cfg_.verbose) {
        std::cerr << "RTSP: worker thread started" << std::endl;
        std::cerr.flush();
    }

    while (running_.load()) {
        gotFirstSample_.store(false, std::memory_order_release);

        if (!buildPipeline()) {
            std::cerr << "RTSP: buildPipeline FAILED" << std::endl;
            std::cerr.flush();
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            continue;
        }

        GstStateChangeReturn sret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
        if (cfg_.verbose) {
            std::cout << "PIPELINE RUNNING (set_state ret=" << sret << ")" << std::endl;
        }

        {
            GstState cur = GST_STATE_NULL, pending = GST_STATE_NULL;
            GstStateChangeReturn gr = gst_element_get_state(pipeline_, &cur, &pending, 2 * GST_SECOND);
            std::cerr << "get_state: ret=" << gr
                      << " cur=" << gst_element_state_get_name(cur)
                      << " pending=" << gst_element_state_get_name(pending)
                      << std::endl;
            std::cerr.flush();
        }

        oneShotNudgeNullToPlayingIfNeeded();

        GstBus* bus = gst_element_get_bus(pipeline_);
        while (running_.load()) {
            GstMessage* msg = gst_bus_timed_pop_filtered(
                bus, 100 * GST_MSECOND,
                (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS)
            );
            if (!msg) continue;

            if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
                GError* err = nullptr;
                gchar* dbg = nullptr;
                gst_message_parse_error(msg, &err, &dbg);
                std::cerr << "GStreamer ERROR: " << (err ? err->message : "unknown") << std::endl;
                if (dbg) std::cerr << "DETAILS: " << dbg << std::endl;
                std::cerr.flush();
                if (err) g_error_free(err);
                if (dbg) g_free(dbg);
                gst_message_unref(msg);
                break;
            }
            if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_EOS) {
                std::cerr << "GStreamer: EOS" << std::endl;
                std::cerr.flush();
                gst_message_unref(msg);
                break;
            }
            gst_message_unref(msg);
        }
        gst_object_unref(bus);

        destroyPipeline();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    destroyPipeline();
    if (cfg_.verbose) {
        std::cerr << "RTSP: worker thread stopped" << std::endl;
        std::cerr.flush();
    }
}

bool RtspWorker::buildPipeline() {
    destroyPipeline();

    pipeline_ = gst_pipeline_new("pipe");
    src_        = gst_element_factory_make("rtspsrc",      "src");
    depay_      = gst_element_factory_make("rtph265depay", "depay");
    parse_      = gst_element_factory_make("h265parse",    "parse");
    dec_        = gst_element_factory_make("mppvideodec",  "dec");
    force_nv12_ = gst_element_factory_make("capsfilter",   "force_nv12");
    sink_       = gst_element_factory_make("appsink",      "mysink");

    if (!pipeline_ || !src_ || !depay_ || !parse_ || !dec_ || !force_nv12_ || !sink_) {
        return false;
    }

    g_object_set(src_,
                 "location", cfg_.url.c_str(),
                 "latency", cfg_.latency_ms,
                 "timeout", (guint64)cfg_.timeout_us,
                 "tcp-timeout", (guint64)cfg_.tcp_timeout_us,
                 "protocols", cfg_.protocols,
                 NULL);

    {
        GstCaps* caps = gst_caps_from_string(cfg_.caps_force.c_str());
        g_object_set(force_nv12_, "caps", caps, NULL);
        gst_caps_unref(caps);
    }

    g_object_set(sink_,
                 "emit-signals", TRUE,
                 "sync", FALSE,
                 "max-buffers", 1,
                 "drop", TRUE,
                 NULL);

    g_signal_connect(sink_, "new-sample", G_CALLBACK(&RtspWorker::onNewSample), this);

    gst_bin_add_many(GST_BIN(pipeline_), src_, depay_, parse_, dec_, force_nv12_, sink_, NULL);

    if (!gst_element_link_many(depay_, parse_, dec_, force_nv12_, sink_, NULL)) {
        std::cerr << "ERROR: Linking depay/parse/dec/force_nv12/sink failed." << std::endl;
        std::cerr.flush();
        return false;
    } else {
        std::cout << "LINK OK: depay->parse->dec->force_nv12->sink" << std::endl;
    }

    g_signal_connect(src_, "pad-added", G_CALLBACK(&RtspWorker::onPadAdded), this);
    return true;
}

void RtspWorker::destroyPipeline() {
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
    }
    pipeline_ = src_ = depay_ = parse_ = dec_ = force_nv12_ = sink_ = nullptr;
}

void RtspWorker::oneShotNudgeNullToPlayingIfNeeded() {
    auto t0 = std::chrono::steady_clock::now();
    while (running_.load()) {
        if (gotFirstSample_.load(std::memory_order_acquire)) return;
        if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(1200)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (!running_.load()) return;
    if (gotFirstSample_.load(std::memory_order_acquire)) return;

    std::cerr << "WARN: no on_new_sample yet -> ONE-SHOT NULL->PLAYING nudge" << std::endl;
    std::cerr.flush();

    gst_element_set_state(pipeline_, GST_STATE_NULL);
    gst_element_set_state(pipeline_, GST_STATE_PLAYING);
}

GstFlowReturn RtspWorker::onNewSample(GstAppSink* appsink, gpointer user_data) {
    auto* self = static_cast<RtspWorker*>(user_data);

//    std::cout << "callback on_new_sample" << std::endl;
    self->gotFirstSample_.store(true, std::memory_order_release);

    GstSample* sample = gst_app_sink_pull_sample(appsink);
    if (!sample) return GST_FLOW_ERROR;

    GstCaps* caps = gst_sample_get_caps(sample);
    GstStructure* s = gst_caps_get_structure(caps, 0);

    gint width = 0, height = 0;
    gst_structure_get_int(s, "width", &width);
    gst_structure_get_int(s, "height", &height);
    const gchar* format = gst_structure_get_string(s, "format");

    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }

    if (format && g_strcmp0(format, "NV12") == 0) {
        cv::Mat yuv(height + height / 2, width, CV_8UC1, (void*)map.data);
        cv::Mat bgr;
        cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12);
        self->store_.setFrame(std::move(bgr));
    }

    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

void RtspWorker::onPadAdded(GstElement* /*src*/, GstPad* new_pad, gpointer user_data) {
    auto* self = static_cast<RtspWorker*>(user_data);

    GstCaps* caps = gst_pad_get_current_caps(new_pad);
    if (!caps) caps = gst_pad_query_caps(new_pad, nullptr);
    if (!caps) return;

    GstStructure* st = gst_caps_get_structure(caps, 0);
    const gchar* encoding = gst_structure_get_string(st, "encoding-name");

    if (!encoding || std::strcmp(encoding, "H265") != 0) {
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
        std::cerr << "[pad-added] LINK RESULT = " << ret << std::endl;
        std::cerr.flush();
    }

    gst_object_unref(sinkpad);
    gst_caps_unref(caps);
}
