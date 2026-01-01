#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#include "tracker_manager.h"      // struct Target { int id; cv::Rect bbox; ... }
#include "static_box_manager.h"   // struct static_box { int id; cv::Rect rect; static_box_state state; ... }
#include "static_target_manager.h"
#include "config.h"

//------------------------------------------------------------------------------
// OverlayRenderer
//
// Draws on-screen overlays:
//  - dynamic boxes (tracker targets)
//  - static boxes (user-locked boxes)
//
// No keyboard handling.
//------------------------------------------------------------------------------
class OverlayRenderer {
public:
    struct Config {
        float hud_alpha;
        float unselected_alpha_when_selected; //TODO
        int dynamic_bbox_window; //TODO
    };

    explicit OverlayRenderer(const toml::table& tbl);

    // Draw dynamic tracked targets.
    // selected_id currently unused for dynamics; kept for future extension.
    void render(
            cv::Mat& frame,
            const std::vector<Target>& targets,
            int selected_id
    ) const ;

    // Draw static user-locked boxes.
    void render_static_boxes(
            cv::Mat& frame,
            const std::vector<static_box>& boxes
    );

    void render_static_targets(
            cv::Mat& frame,
            const std::vector<StaticTarget>& targets
    ) const;

private:
    Config cfg_;
    bool load_overlay_config(const toml::table& tbl);

    static cv::Rect clip_rect(const cv::Rect& r, int w, int h);

    static void draw_rect_alpha(
            cv::Mat& frame,
            const cv::Rect& r,
            const cv::Scalar& color,
            float alpha
    );

    static void draw_label(
            cv::Mat& frame,
            const cv::Point& org,
            const std::string& text,
            const cv::Scalar& color
    );
};
