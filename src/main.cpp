#include <gst/gst.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include "frame_store.h"
#include "rtsp_worker.h"
#include "manual_tracker_manager.h"
#include "overlay.h"
#include <atomic>
#include <mutex>
#include <thread>
#include "display_loop.h"
#include "other_handlers.h"
#include "rtsp_watch_dog.h"

#define SHOW_IDS = false;


std::atomic<bool> g_running{true};
std::atomic<bool> g_rtsp_restart_requested{false};

// Момент времени (в миллисекундах steady_clock) когда мы последний раз
// получили кадр от RTSP. Используется watchdog'ом.
static std::atomic<long long> g_last_frame_ms{0};

// Текущее время по steady_clock в миллисекундах (монотонное, без скачков времени).
static inline long long now_steady_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}



// ===================== GLOBALS FOR MOUSE CALLBACK =====================

static ManualTrackerManager *g_manual_tracker = nullptr;
static std::mutex g_last_frame_mutex;
static cv::Mat g_last_frame;


// ===================== MOUSE CALLBACK =====================

void on_mouse(int event, int x, int y, int, void *) {
    if (event != cv::EVENT_LBUTTONDOWN)
        return;

    if (!g_manual_tracker)
        return;

    std::lock_guard<std::mutex> lock(g_last_frame_mutex);
    if (g_last_frame.empty()) {
        return;
    }
    g_manual_tracker->handle_click(x, y, g_last_frame, now_steady_ms());
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
    RtspWatchDog rtsp_watchdog;


    // получаем конфигураци из config.toml
    toml::table tbl = toml::parse_file("config.toml");
    RtspWorker rtsp(raw_store, tbl);
    load_rtsp_watchdog(tbl, rtsp_watchdog);
    ManualTrackerManager tracker(tbl);
    OverlayRenderer overlay(tbl);
    rtsp.start();



    const long long app_start_ms = now_steady_ms();
    g_last_frame_ms.store(app_start_ms, std::memory_order_relaxed);

    std::atomic<long long> last_restart_ms{app_start_ms - 100000};

    std::thread control_thread([&]() {
        std::cout << "[CTRL] Control thread started" << std::endl;
        while (g_running.load(std::memory_order_relaxed)) {

            // Ручной рестарт по R / r
            if (g_rtsp_restart_requested.exchange(false, std::memory_order_acq_rel)) {
                std::cout << "[CTRL] Manual RTSP restart" << std::endl;
                rtsp.stop();
                rtsp.start();
                last_restart_ms.store(now_steady_ms(), std::memory_order_relaxed);
                g_last_frame_ms.store(now_steady_ms(), std::memory_order_relaxed);
            }

            auto rtsp_watchdog_tick = [&](){
                // Watchdog: если давно нет кадров — перезапускаем RTSP
                const long long now_ms = now_steady_ms();
                const long long since_start = now_ms - app_start_ms;
                const long long no_frame_ms = now_ms - g_last_frame_ms.load(std::memory_order_relaxed);
                const long long since_restart = now_ms - last_restart_ms.load(std::memory_order_relaxed);


                // =====================================================================
                // Control-поток: watchdog RTSP + ручной рестарт по клавише R.
                // Здесь выполняются тяжёлые операции stop/start RTSP, чтобы НЕ блокировать UI.
                // =====================================================================
                static const int WD_NO_FRAME_TIMEOUT_MS = rtsp_watchdog.no_frame_timeout_ms
                                                          ? rtsp_watchdog.no_frame_timeout_ms : 1500;
                static const int WD_RESTART_COOLDOWN_MS = rtsp_watchdog.restart_cooldown_ms
                                                          ? rtsp_watchdog.restart_cooldown_ms : 1000;
                static const int WD_STARTUP_GRACE_MS = rtsp_watchdog.startup_grace_ms ? rtsp_watchdog.startup_grace_ms
                                                                                      : 1500;

                if (since_start > WD_STARTUP_GRACE_MS &&
                    no_frame_ms > WD_NO_FRAME_TIMEOUT_MS &&
                    since_restart > WD_RESTART_COOLDOWN_MS) {

                    std::cout << "[WATCHDOG] No frames for " << no_frame_ms
                              << " ms -> restarting RTSP" << std::endl;

                    rtsp.stop();
                    rtsp.start();
                    last_restart_ms.store(now_ms, std::memory_order_relaxed);
                    g_last_frame_ms.store(now_ms, std::memory_order_relaxed);
                }
            };
            rtsp_watchdog_tick();

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::cout << "[CTRL] Control thread exit" << std::endl;
    });

    g_manual_tracker = &tracker;


    std::thread tracker_thread([&]() {
        cv::Mat frame;
        while (g_running.load(std::memory_order_relaxed)) {

            if (!raw_store.waitFrame(frame, 100)) {
                continue;
            }

            // Кадр получен: обновляем таймер watchdog-а.
            g_last_frame_ms.store(now_steady_ms(), std::memory_order_relaxed);

            {
                std::lock_guard<std::mutex> lock(g_last_frame_mutex);
                g_last_frame = frame.clone();
            }

            tracker.update(frame, now_steady_ms());

            overlay.render(frame, tracker.targets(), -1);

            // Публикуем кадр с overlay для UI (отдельный store => нет "саморазгона")
            ui_store.setFrame(std::move(frame));
        }
        std::cout << "[TRK] tracker thread exit" << std::endl;
    });

    DisplayLoop display_loop(ui_store);
    display_loop.run();   // ЕДИНСТВЕННЫЙ UI-LOOP

    if (tracker_thread.joinable()) tracker_thread.join();


// Запрос на остановку RTSP/потоков уже выставлен через g_running.
    // Аккуратно завершаем RTSP.
    rtsp.stop();

    // Будим возможные ожидания по кадрам.
    raw_store.stop();
    ui_store.stop();

    // Дожидаемся control-потока, чтобы он не работал с rtsp после уничтожения.
    if (control_thread.joinable()) control_thread.join();

    return 0;
}
