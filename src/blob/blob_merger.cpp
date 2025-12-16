#include "blob/blob_merger.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace blob {

    static float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
        cv::Rect2f inter = a & b;
        float ia = inter.area();
        if (ia <= 0.f) return 0.f;
        float ua = a.area() + b.area() - ia;
        return ua > 0.f ? ia / ua : 0.f;
    }

    static cv::Point2f centerOf(const cv::Rect2f& r) {
        return { r.x + r.width*0.5f, r.y + r.height*0.5f };
    }

    static float dist(const cv::Point2f& a, const cv::Point2f& b) {
        float dx = a.x - b.x;
        float dy = a.y - b.y;
        return std::sqrt(dx*dx + dy*dy);
    }

    static cv::Rect2f expand(const cv::Rect2f& r, float factor, int gap) {
        float ex = r.width  * factor + (float)gap;
        float ey = r.height * factor + (float)gap;
        return cv::Rect2f(r.x - ex, r.y - ey, r.width + 2*ex, r.height + 2*ey);
    }

    struct DSU {
        std::vector<int> p, r;
        explicit DSU(int n) : p(n), r(n,0) { std::iota(p.begin(), p.end(), 0); }
        int find(int x){ return p[x]==x?x:p[x]=find(p[x]); }
        void uni(int a,int b){
            a=find(a); b=find(b);
            if(a==b) return;
            if(r[a]<r[b]) std::swap(a,b);
            p[b]=a;
            if(r[a]==r[b]) r[a]++;
        }
    };

    std::vector<cv::Rect2f> merge_blobs(const std::vector<cv::Rect2f>& in,
                                        const cv::Size& frame_size,
                                        const MergeParams& p)
    {
        std::vector<cv::Rect2f> v;
        v.reserve(in.size());

        // clamp + filter tiny
        for (auto r : in) {
            cv::Rect2f fr(0,0,(float)frame_size.width,(float)frame_size.height);
            r = r & fr;
            if (r.area() >= (float)p.min_area_px) v.push_back(r);
        }
        if (v.empty()) return {};

        const int n = (int)v.size();
        DSU dsu(n);

        // union by overlap/near
        for (int i = 0; i < n; ++i) {
            cv::Rect2f ei = expand(v[i], p.expand_factor, p.gap_px);
            cv::Point2f ci = centerOf(v[i]);

            for (int j = i+1; j < n; ++j) {
                cv::Rect2f ej = expand(v[j], p.expand_factor, p.gap_px);
                cv::Point2f cj = centerOf(v[j]);

                bool close = dist(ci, cj) <= p.center_dist_px;
                bool ov_iou = iou(v[i], v[j]) >= p.iou_threshold;
                bool ov_exp = (ei & ej).area() > 0.f;

                if (ov_iou || ov_exp || close) dsu.uni(i, j);
            }
        }

        // collect groups
        std::vector<cv::Rect2f> out;
        out.reserve(n);

        std::vector<int> root(n);
        for (int i = 0; i < n; ++i) root[i] = dsu.find(i);

        // map root -> bbox union
        std::vector<int> uniq;
        uniq.reserve(n);
        for (int i = 0; i < n; ++i) uniq.push_back(root[i]);
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());

        for (int r0 : uniq) {
            cv::Rect2f u = v[0];
            bool first = true;
            for (int i = 0; i < n; ++i) {
                if (root[i] != r0) continue;
                if (first) { u = v[i]; first = false; }
                else {
                    float x1 = std::min(u.x, v[i].x);
                    float y1 = std::min(u.y, v[i].y);
                    float x2 = std::max(u.x + u.width,  v[i].x + v[i].width);
                    float y2 = std::max(u.y + u.height, v[i].y + v[i].height);
                    u = cv::Rect2f(x1, y1, x2-x1, y2-y1);
                }
            }
            out.push_back(u);
        }

        // sort by area desc, keep top max_out
        std::sort(out.begin(), out.end(),
                  [](const cv::Rect2f& a, const cv::Rect2f& b){ return a.area() > b.area(); });

        if ((int)out.size() > p.max_out) out.resize((size_t)p.max_out);
        return out;
    }

} // namespace blob
