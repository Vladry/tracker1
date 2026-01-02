#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

struct ManualMotionDetectorConfig {
    int click_capture_size = 80;
    int motion_frames = 3;
    int motion_diff_threshold = 25;
    int click_padding = 6;
    int tracker_init_padding = 10;
    int tracker_min_size = 24;
    float motion_min_magnitude = 0.4f;
    float motion_angle_tolerance_deg = 20.0f;
    float motion_mag_tolerance_px = 3.0f;
    int min_area = 200;
    int min_width = 10;
    int min_height = 10;
};

class ManualMotionDetector {
public:
    ManualMotionDetector() = default;
    explicit ManualMotionDetector(const ManualMotionDetectorConfig& cfg) : cfg_(cfg) {}

    void update_config(const ManualMotionDetectorConfig& cfg) { cfg_ = cfg; }
    int required_frames() const;

    cv::Rect make_click_roi(const cv::Mat& frame, int x, int y) const;

    bool build_candidate(const std::vector<cv::Mat>& gray_frames,
                         const cv::Rect& roi,
                         const cv::Size& frame_size,
                         cv::Rect2f& out_tracker_roi,
                         std::vector<cv::Point2f>* motion_points,
                         cv::Rect2f* motion_roi_out) const;

private:
    ManualMotionDetectorConfig cfg_{};

    cv::Rect2f build_motion_roi_from_sequence(const std::vector<cv::Mat>& frames,
                                              const cv::Rect& roi,
                                              std::vector<cv::Point2f>& motion_points) const;
    cv::Rect2f build_motion_roi_from_diff(const std::vector<cv::Mat>& frames,
                                          const cv::Rect& roi) const;
};
