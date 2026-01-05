#include <gst/gst.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <filesystem>
#include <vector>
#include "frame_store.h"
#include "rtsp_worker.h"
#include "manual_tracker_manager.h"
#include "static_target_manager.h"
#include "overlay.h"
#include <atomic>
#include <mutex>
#include <thread>
#include "display_loop.h"
#include "other_handlers.h"
#include "config.h"

#define SHOW_IDS = false;


std::atomic<bool> g_running{true};
std::atomic<bool> g_rtsp_restart_requested{false};

// Текущее время по steady_clock в миллисекундах (монотонное, без скачков времени).
static inline long long now_steady_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Загружает и объединяет конфигурационные таблицы из TOML-файлов.
static toml::table load_config_tables() {
    toml::table combined;
    auto merge_table = [&](const toml::table& src, std::string_view key) {
        const auto* table = src.get_as<toml::table>(key);
        if (!table) {
            throw std::runtime_error(std::string("missing [") + std::string(key) + "] table");
        }
        combined.insert_or_assign(std::string(key), *table);
    };
    auto merge_optional_file = [&](const std::string& path, std::string_view key) {
        if (!std::filesystem::exists(path)) {
            std::cout << "[CONFIG] optional file missing: " << path << std::endl;
            return;
        }
        try {
            const toml::table tbl = toml::parse_file(path);
            const auto* table = tbl.get_as<toml::table>(key);
            if (!table) {
                std::cout << "[CONFIG] optional table missing: [" << key << "] in " << path << std::endl;
                return;
            }
            combined.insert_or_assign(std::string(key), *table);
        } catch (const toml::parse_error& err) {
            std::cout << "[CONFIG] optional file parse failed: " << path
                      << " error=" << err.description() << std::endl;
        } catch (const std::exception& err) {
            std::cout << "[CONFIG] optional file parse failed: " << path
                      << " error=" << err.what() << std::endl;
        }
    };


    const toml::table trackers = toml::parse_file("trackers.toml");
    merge_table(trackers, "tracker");
    merge_table(trackers, "manual_tracker");

    const toml::table static_detector = toml::parse_file("static_detector.toml");
    merge_table(static_detector, "static_detector");

    const toml::table logging = toml::parse_file("logging.toml");
    merge_table(logging, "logging");

    const toml::table overlay = toml::parse_file("overlay.toml");
    merge_table(overlay, "overlay");

    merge_optional_file("display.toml", "display");

    const toml::table rebind_smoothing = toml::parse_file("rebind_smoothing.toml");
    merge_table(rebind_smoothing, "static_rebind");
    merge_table(rebind_smoothing, "smoothing");

    const toml::table rtsp = toml::parse_file("rtsp.toml");
    merge_table(rtsp, "rtsp");

    return combined;
}


// ===================== GLOBALS FOR MOUSE CALLBACK =====================

static ManualTrackerManager *g_manual_tracker = nullptr;
static StaticTargetManager *g_static_manager = nullptr;
static std::mutex g_last_frame_mutex;
static cv::Mat g_last_frame;
static LoggingConfig g_logging;


// ===================== MOUSE CALLBACK =====================

// Обрабатывает клики мыши и перенаправляет в ручные менеджеры.
void on_mouse(int event, int x, int y, int, void *) {
    if (!g_manual_tracker && !g_static_manager) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_last_frame_mutex);
    if (g_last_frame.empty()) {
        if (g_logging.MOUSE_CLICK_LOGGER) {
            std::cout << "[MOUSE] click x=" << x << " y=" << y << " ignored empty frame" << std::endl;
        }
        return;
    }
    bool handled = false;
    if (event == cv::EVENT_LBUTTONDOWN && g_manual_tracker) {
        handled = g_manual_tracker->handle_click(x, y, g_last_frame, now_steady_ms());
    } else if (event == cv::EVENT_RBUTTONDOWN && g_static_manager) {
        handled = g_static_manager->handle_right_click(x, y, g_last_frame, now_steady_ms());
    } else {
        return;
    }
    if (g_logging.MOUSE_CLICK_LOGGER) {
        std::cout << "[MOUSE] click x=" << x << " y=" << y
                  << " button=" << ((event == cv::EVENT_RBUTTONDOWN) ? "RMB" : "LMB")
                  << " handled=" << (handled ? "true" : "false") << std::endl;
    }
}




int main(int argc, char *argv[]) {

    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    if (!std::getenv("DISPLAY")) {
        setenv("DISPLAY", ":0", 0);
    }

    gst_init(&argc, &argv);
    std::cout << "STARTING PIPELINE..." << std::endl;

    FrameStore raw_store;
    FrameStore ui_store;


    // получаем конфигурации из всех файлов.toml
    toml::table tbl = load_config_tables();
    load_logging_config(tbl, g_logging);
    RtspWorker rtsp(raw_store, tbl);
    ManualTrackerManager tracker(tbl);
    StaticTargetManager static_targets(tbl);
    OverlayRenderer overlay(tbl);
    rtsp.start();



    std::thread control_thread([&]() {
        std::cout << "[CTRL] Control thread started" << std::endl;
        while (g_running.load(std::memory_order_relaxed)) {

            // Ручной рестарт по R / r
            if (g_rtsp_restart_requested.exchange(false, std::memory_order_acq_rel)) {
                std::cout << "[CTRL] Manual RTSP restart" << std::endl;
                rtsp.stop();
                rtsp.start();
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::cout << "[CTRL] Control thread exit" << std::endl;
    });

    g_manual_tracker = &tracker;
    g_static_manager = &static_targets;


    std::thread tracker_thread([&]() {
        cv::Mat frame;
        while (g_running.load(std::memory_order_relaxed)) {

            if (!raw_store.waitFrame(frame)) {
                continue;
            }

            {
                std::lock_guard<std::mutex> lock(g_last_frame_mutex);
                g_last_frame = frame.clone();
            }

            tracker.update(frame, now_steady_ms());
            overlay.render(frame, tracker.targets(), -1);
            static_targets.update(frame, now_steady_ms());
            overlay.render_static_targets(frame, static_targets.targets());

            // Пуликуем кадр с overlay для UI (отдельный store => нет "саморазгона")
            ui_store.setFrame(std::move(frame));
        }
        std::cout << "[TRK] tracker thread exit" << std::endl;
    });

    DisplayLoop::Config display_cfg;
    const auto* display = tbl["display"].as_table();
    if (display) {
        display_cfg.TARGET_FPS = read_required<int>(*display, "TARGET_FPS");
        display_cfg.WINDOW_NAME = read_required<std::string>(*display, "WINDOW_NAME");
    } else {
        std::cout << "[DISPLAY] using default settings (missing [display])" << std::endl;
    }
    DisplayLoop display_loop(ui_store, display_cfg);
    display_loop.run();   // ЕДИНСТВЕННЫЙ UI-LOOP

    // Будим возможные ожидания по кадрам.
    raw_store.stop();
    ui_store.stop();

    if (tracker_thread.joinable()) tracker_thread.join();


    // Запрос на остановку RTSP/пооков уже выставлен через g_running.
    // Аккуратно завершаем RTSP.
    rtsp.stop();

    // Дожидаемся control-потока, чтобы он не работал с rtsp после уничтожения.
    if (control_thread.joinable()) control_thread.join();

    return 0;
}
