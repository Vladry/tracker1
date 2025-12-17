#include "overlay.h"
#include <algorithm>

OverlayRenderer::OverlayRenderer(const Config& cfg) : cfg_(cfg) {}

static void putKeyVal(cv::Mat& img, int x, int y, const std::string& k, const std::string& v) {
    double fs = 0.55;
    int th = 1;
    cv::putText(img, k, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, fs, cv::Scalar(0,255,0), th, cv::LINE_AA);
    cv::putText(img, v, cv::Point(x + 170, y), cv::FONT_HERSHEY_SIMPLEX, fs, cv::Scalar(0,0,255), th, cv::LINE_AA);
}

void OverlayRenderer::drawCornerTables(cv::Mat& overlay, const Target* sel) {
    const int pad = 12;
    const int row = 18;
    const int boxw = 340;
    const int boxh = 9*row + 2*pad;

    int W = overlay.cols, H = overlay.rows;
    std::vector<cv::Point> corners = {
        {pad, pad},
        {W - boxw - pad, pad},
        {pad, H - boxh - pad},
        {W - boxw - pad, H - boxh - pad},
    };

    for (auto p : corners) {
        cv::Rect r(p.x, p.y, boxw, boxh);
        cv::rectangle(overlay, r, cv::Scalar(0,0,0), -1, cv::LINE_AA);
        cv::rectangle(overlay, r, cv::Scalar(255,255,255), 1, cv::LINE_AA);

        int x = p.x + pad;
        int y = p.y + pad + row;

        double az = sel ? sel->azimuth_deg : 0.0;
        double el = sel ? sel->elevation_deg : 0.0;
        double dist = sel ? sel->distance_m : 0.0;
        int id = sel ? sel->id : 0;

        putKeyVal(overlay, x, y + 0*row,  "ID", std::to_string(id));
        putKeyVal(overlay, x, y + 1*row,  "Azimuth (deg)", cv::format("%.2f", az));
        putKeyVal(overlay, x, y + 2*row,  "Elevation (deg)", cv::format("%.2f", el));
        putKeyVal(overlay, x, y + 3*row,  "Distance (m)", cv::format("%.2f", dist));
        putKeyVal(overlay, x, y + 4*row,  "SpeedX", cv::format("%.2f", 0.0));
        putKeyVal(overlay, x, y + 5*row,  "SpeedY", cv::format("%.2f", 0.0));
        putKeyVal(overlay, x, y + 6*row,  "dAz/dt", cv::format("%.2f", 0.0));
        putKeyVal(overlay, x, y + 7*row,  "dEl/dt", cv::format("%.2f", 0.0));
    }
}

void OverlayRenderer::drawTargetBoxBlended(cv::Mat& frame_bgr,
                                          const Target& t,
                                          bool is_selected,
                                          bool selection_exists) {
    if (frame_bgr.empty()) return;

    cv::Mat overlay = frame_bgr.clone();

    cv::Scalar col = is_selected ? cv::Scalar(0,0,255) : cv::Scalar(0,255,0);
    cv::rectangle(overlay, t.bbox, col, 2, cv::LINE_AA);

    std::string label = "ID=" + std::to_string(t.id);
    cv::putText(overlay, label,
                cv::Point((int)t.bbox.x, (int)std::max(0.f, t.bbox.y - 4.f)),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv::LINE_AA);

    float a = 1.0f;
    if (selection_exists && !is_selected) {
        a = std::min(1.f, std::max(0.f, cfg_.unselected_alpha_when_selected));
    }
    cv::addWeighted(overlay, a, frame_bgr, 1.f - a, 0.0, frame_bgr);
}

void OverlayRenderer::render(cv::Mat& frame_bgr,
                            const std::vector<Target>& targets,
                            int selected_id) {
    if (frame_bgr.empty()) return;

    const Target* selected = nullptr;
    for (const auto& t : targets) if (t.id == selected_id) { selected = &t; break; }
    const bool selection_exists = (selected != nullptr);

    for (const auto& t : targets) {
        drawTargetBoxBlended(frame_bgr, t, t.id == selected_id, selection_exists);
    }

    if (cfg_.hud_alpha > 0.0f) {
        cv::Mat hud = frame_bgr.clone();
        drawCornerTables(hud, selected);
        float a = std::min(1.f, std::max(0.f, cfg_.hud_alpha));
        cv::addWeighted(hud, a, frame_bgr, 1.f - a, 0.0, frame_bgr);
    }
}
