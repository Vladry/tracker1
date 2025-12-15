#pragma once
#include <string>
#include <opencv2/opencv.hpp>

enum class TargetClass : int {
    Unknown = 0
};

struct TargetSize {
    double width_m  = 0.0;
    double height_m = 0.0;
};

struct Target {
    int id = -1;
    std::string target_name;
    TargetClass target_class = TargetClass::Unknown;
    TargetSize target_size;

    // angles relative to camera optical axis (placeholder for now)
    double azimuth_deg = 0.0;          // азимут
    double elevation_deg = 0.0;        // угол возвышения

    double distance_m = 0.0;
    double speedX_mps = 0.0;
    double speedY_mps = 0.0;

    cv::Rect2f bbox;
    int age_frames = 0;
    int missed_frames = 0;
};
