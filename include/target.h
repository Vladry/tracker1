#pragma once
#include <string>
#include <opencv2/opencv.hpp>

enum class TargetClass : int {
    Unknown = 0
};

struct TargetSize {
    double width_m  = 0.0; // - ширина цели (метры).
    double height_m = 0.0; // - высота цели (метры).
};

struct Target {
    int id = -1; // - идентификатор цели.
    std::string target_name; // - имя/метка цели.
    TargetClass target_class = TargetClass::Unknown; // - класс цели.
    TargetSize target_size; // - физические размеры цели.

    // angles relative to camera optical axis (placeholder for now)
    double azimuth_deg = 0.0; // - азимут цели относительно оси камеры.
    double elevation_deg = 0.0; // - угол возвышения цели относительно оси камеры.

    double distance_m = 0.0; // - дистанция до цели (метры).
    double speedX_mps = 0.0; // - скорость по X (м/с).
    double speedY_mps = 0.0; // - скорость по Y (м/с).

    cv::Rect2f bbox; // - прямоугольник цели в кадре.
    bool has_cross = false; // - нужно ли рисовать прицел для цели.
    cv::Point2f cross_center{0.0f, 0.0f}; // - центр красного крестика.
    int missed_frames = 0; // - количество пропущенных кадров.
};
