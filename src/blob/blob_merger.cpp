#include "blob/blob_merger.h"
#include "util/rect_utils.h"
#include <numeric>
#include <algorithm>

namespace {

struct DSU {
    std::vector<int> p, r;
    explicit DSU(int n): p(n), r(n,0) { std::iota(p.begin(), p.end(), 0); }
    int find(int a){ return p[a]==a ? a : (p[a]=find(p[a])); }
    void unite(int a,int b){
        a=find(a); b=find(b);
        if(a==b) return;
        if(r[a]<r[b]) std::swap(a,b);
        p[b]=a;
        if(r[a]==r[b]) r[a]++;
    }
};

static cv::Rect2f rect_union(const cv::Rect2f& a, const cv::Rect2f& b) {
    float x1 = std::min(a.x, b.x);
    float y1 = std::min(a.y, b.y);
    float x2 = std::max(a.x + a.width,  b.x + b.width);
    float y2 = std::max(a.y + a.height, b.y + b.height);
    return cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
}

// minimal distance between rectangles (0 if overlaps)
static bool close_enough(const cv::Rect2f& a, const cv::Rect2f& b, int gap_px) {
    float ax2 = a.x + a.width;
    float ay2 = a.y + a.height;
    float bx2 = b.x + b.width;
    float by2 = b.y + b.height;

    float dx = 0.0f;
    if (ax2 < b.x) dx = b.x - ax2;
    else if (bx2 < a.x) dx = a.x - bx2;

    float dy = 0.0f;
    if (ay2 < b.y) dy = b.y - ay2;
    else if (by2 < a.y) dy = a.y - by2;

    return (dx <= (float)gap_px) && (dy <= (float)gap_px);
}

} // namespace

namespace blob {

std::vector<cv::Rect2f> merge_blobs(const std::vector<cv::Rect2f>& in,
                                   const cv::Size& frame_size,
                                   const MergeParams& p)
{
    if (in.empty()) return {};

    // Expand slightly to encourage merging of sub-blobs
    std::vector<cv::Rect2f> r(in.size());
    for (size_t i=0;i<in.size();++i) {
        r[i] = util::expandAndClamp(in[i], frame_size, p.expand_factor);
    }

    DSU dsu((int)r.size());

    for (int i=0;i<(int)r.size();++i) {
        for (int j=i+1;j<(int)r.size();++j) {
            float v = util::iou(r[i], r[j]);
            if (v >= p.iou_threshold || close_enough(r[i], r[j], p.gap_px)) {
                dsu.unite(i, j);
            }
        }
    }

    std::vector<cv::Rect2f> merged(r.size());
    std::vector<bool> has(r.size(), false);
    for (int i=0;i<(int)r.size();++i) {
        int root = dsu.find(i);
        if (!has[root]) { merged[root] = r[i]; has[root] = true; }
        else merged[root] = rect_union(merged[root], r[i]);
    }

    std::vector<cv::Rect2f> out;
    out.reserve(r.size());
    for (int i=0;i<(int)r.size();++i) {
        if (has[i]) out.push_back(util::clampRect(merged[i], frame_size));
    }

    // sort by area descending (useful when later enforcing max targets)
    std::sort(out.begin(), out.end(), [](const cv::Rect2f& a, const cv::Rect2f& b){
        return a.area() > b.area();
    });

    return out;
}

} // namespace blob
