#pragma once
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <atomic>
#include <string>
#include <thread>
#include "frame_store.h"

class RtspWorker {
public:
    struct Config {
        std::string url;
        int protocols = 1;              // 1 = UDP
        int latency_ms = 0;
        guint64 timeout_us = 2000000;   // 2s
        guint64 tcp_timeout_us = 2000000;
        std::string caps_force = "video/x-raw,format=NV12";
        bool verbose = true;
    };

    RtspWorker(FrameStore& store, const Config& cfg);
    ~RtspWorker();

    void start();
    void stop();

private:
    void threadMain();
    bool buildPipeline();
    void destroyPipeline();

    void oneShotNudgeNullToPlayingIfNeeded();

    static GstFlowReturn onNewSample(GstAppSink* appsink, gpointer user_data);
    static void onPadAdded(GstElement* src, GstPad* new_pad, gpointer user_data);

private:
    FrameStore& store_;
    Config cfg_;

    std::atomic<bool> running_{false};
    std::atomic<bool> gotFirstSample_{false};
    std::thread th_;

    GstElement* pipeline_{nullptr};
    GstElement* src_{nullptr};
    GstElement* depay_{nullptr};
    GstElement* parse_{nullptr};
    GstElement* dec_{nullptr};
    GstElement* force_nv12_{nullptr};
    GstElement* sink_{nullptr};
};
