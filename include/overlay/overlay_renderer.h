#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "tracker/target.h"

class OverlayRenderer {
public:
    struct Config {
        float alpha = 0.35f; // 0..1
        bool hide_others_when_selected = true;
    };

    explicit OverlayRenderer(const Config& cfg);

    void render(cv::Mat& frame_bgr,
                const std::vector<Target>& targets,
                int selected_id);

private:
    Config cfg_;
    void drawCornerTables(cv::Mat& overlay, const Target* sel);
};
