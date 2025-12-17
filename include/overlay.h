#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "target.h"

// Renders targets + HUD tables.
// Requirement:
// - When a target is selected (red), other targets stay visible but with configurable transparency.
// - When no target is selected, all targets are fully visible.
class OverlayRenderer {
public:
    struct Config {
        float hud_alpha = 0.35f;                     // transparency of HUD tables (0..1)
        float unselected_alpha_when_selected = 0.1f; // transparency for non-selected boxes when selection exists
    };

    explicit OverlayRenderer(const Config& cfg);

    void render(cv::Mat& frame_bgr,
                const std::vector<Target>& targets,
                int selected_id);

private:
    Config cfg_;
    void drawCornerTables(cv::Mat& overlay, const Target* sel);
    void drawTargetBoxBlended(cv::Mat& frame_bgr, const Target& t, bool is_selected, bool selection_exists);
};
