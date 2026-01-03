#pragma once
#include <string>
#include <opencv2/opencv.hpp>

enum class TargetClass : int {
    Unknown = 0
};

struct TargetSize {
    double width_m  = 0.0; // - width_m: ширина цели (метры).
    double height_m = 0.0; // - height_m: высота цели (метры).
};

struct Target {
    int id = -1; // - id: идентификатор цели.
    std::string target_name; // - target_name: имя/метка цели.
    TargetClass target_class = TargetClass::Unknown; // - target_class: класс цели.
    TargetSize target_size; // - target_size: физические размеры цели.

    // angles relative to camera optical axis (placeholder for now)
    double azimuth_deg = 0.0; // - azimuth_deg: азимут цели относительно оси камеры.
    double elevation_deg = 0.0; // - elevation_deg: угол возвышения цели относительно оси камеры.

    double distance_m = 0.0; // - distance_m: дистанция до цели (метры).
    double speedX_mps = 0.0; // - speedX_mps: скорость по X (м/с).
    double speedY_mps = 0.0; // - speedY_mps: скорость по Y (м/с).

    cv::Rect2f bbox; // - bbox: прямоугольник цели в кадре.
    int missed_frames = 0; // - missed_frames: количество пропущенных кадров.
};
