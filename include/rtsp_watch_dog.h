#pragma once

#include "config.h"

struct RtspWatchDog {
    // ------------------------- [rtsp.watchdog] -------------------------
    // Таймаут отсутствия кадров (мс)
    std::uint64_t no_frame_timeout_ms = 1500;

    // Минимальный интервал между рестартами (мс)
    std::uint64_t restart_cooldown_ms = 1000;

    // Льготный период после старта (мс)
    std::uint64_t startup_grace_ms = 3000;

};


bool load_rtsp_watchdog (toml::table &tbl, RtspWatchDog& rtsp_wd);
