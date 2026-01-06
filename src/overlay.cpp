#include "overlay.h"
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

//==============================================================================
// Визуальное сглаживание динамических bbox (рамок)
//
// ВАЖНО:
//  - Это влияет ТОЛЬКО на отображение (рендер).
//  - Логика трекера/детектора не меняется.
//  - Для каждого динамического Target.id хранится свя история.
//==============================================================================

// Состояние сглаживания для одного объекта:
// history хранит несколько последних прямоугольников в float-точности.
struct smooth_rect_state {
    std::deque<cv::Rect2f> history;
};

// Глобальное (в рамках файла) хранилище истории:
// ключ = Target.id, значение = история его bbox.
static std::unordered_map<int, smooth_rect_state> g_dyn_smooth;

//------------------------------------------------------------------------------
// Сглаживание bbox для отображения (скользящее среднее по последним N кадрам)
//------------------------------------------------------------------------------
static cv::Rect smooth_bbox_for_render(int id, const cv::Rect& current, int window_size) {
    // Достаём (или создаём) историю для данного id.
    auto& st = g_dyn_smooth[id];

    // Кладём текущий bbox в историю (в float-формате для точности).
    st.history.emplace_back(cv::Rect2f(current));

    // Поддерживаем ограничение на длину истории.
    const int history_window = std::max(1, window_size);
    if ((int)st.history.size() > history_window)
        st.history.pop_front();

    // Считаем среднее значение по всем bbox в истории.
    cv::Rect2f acc(0, 0, 0, 0);
    for (const auto& r : st.history) {
        acc.x += r.x;
        acc.y += r.y;
        acc.width  += r.width;
        acc.height += r.height;
    }

    // Нормализуем сумму -> среднее.
    float inv = 1.0f / (float)st.history.size();
    acc.x      *= inv;
    acc.y      *= inv;
    acc.width  *= inv;
    acc.height *= inv;

    // Приводим к int (OpenCV Rect с int), округляя.
    int x = (int)std::lround(acc.x);
    int y = (int)std::lround(acc.y);
    int w = (int)std::lround(acc.width);
    int h = (int)std::lround(acc.height);

    // Защита от нулевых/отрицательных размеров.
    if (w < 1) w = 1;
    if (h < 1) h = 1;

    return cv::Rect(x, y, w, h);
}

//------------------------------------------------------------------------------
// Рисование пунктирной линии
//------------------------------------------------------------------------------
static void draw_dashed_line(
        cv::Mat& frame,
        const cv::Point& p1,
        const cv::Point& p2,
        const cv::Scalar& color,
        int thickness,
        int dash_len,
        int gap_len
) {
    const float length = std::hypot(static_cast<float>(p2.x - p1.x),
                                    static_cast<float>(p2.y - p1.y));
    if (length <= 1.0f) {
        return;
    }
    const cv::Point2f direction((p2.x - p1.x) / length, (p2.y - p1.y) / length);
    float dist = 0.0f;
    while (dist < length) {
        const float seg_len = std::min(static_cast<float>(dash_len), length - dist);
        cv::Point start(static_cast<int>(p1.x + direction.x * dist),
                        static_cast<int>(p1.y + direction.y * dist));
        cv::Point end(static_cast<int>(p1.x + direction.x * (dist + seg_len)),
                      static_cast<int>(p1.y + direction.y * (dist + seg_len)));
        cv::line(frame, start, end, color, thickness, cv::LINE_AA);
        dist += static_cast<float>(dash_len + gap_len);
    }
}

static void draw_dashed_rect(
        cv::Mat& frame,
        const cv::Rect& rect,
        const cv::Scalar& color,
        int thickness
) {
    // Рисует пунктирную рамку по периметру прямоугольника.
    const int dash_len = 6;
    const int gap_len = 4;
    const cv::Point p1(rect.x, rect.y);
    const cv::Point p2(rect.x + rect.width, rect.y);
    const cv::Point p3(rect.x + rect.width, rect.y + rect.height);
    const cv::Point p4(rect.x, rect.y + rect.height);
    draw_dashed_line(frame, p1, p2, color, thickness, dash_len, gap_len);
    draw_dashed_line(frame, p2, p3, color, thickness, dash_len, gap_len);
    draw_dashed_line(frame, p3, p4, color, thickness, dash_len, gap_len);
    draw_dashed_line(frame, p4, p1, color, thickness, dash_len, gap_len);
}

static void draw_cross(
        cv::Mat& frame,
        const cv::Point& center,
        int half_size,
        const cv::Scalar& color,
        int thickness
) {
    cv::Point left(center.x - half_size, center.y);
    cv::Point right(center.x + half_size, center.y);
    cv::Point top(center.x, center.y - half_size);
    cv::Point bottom(center.x, center.y + half_size);
    cv::line(frame, left, right, color, thickness, cv::LINE_AA);
    cv::line(frame, top, bottom, color, thickness, cv::LINE_AA);
}

//------------------------------------------------------------------------------
// Конструктор
//------------------------------------------------------------------------------
OverlayRenderer::OverlayRenderer(const toml::table& tbl){
    load_overlay_config(tbl);
}

//------------------------------------------------------------------------------
// Вспомогательные утилиты
//------------------------------------------------------------------------------
cv::Rect OverlayRenderer::clip_rect(const cv::Rect& r, int w, int h) {
    // Ограничиваем координаты прямоугольника рамками кадра.
    int x1 = std::max(0, r.x);
    int y1 = std::max(0, r.y);
    int x2 = std::min(w, r.x + r.width);
    int y2 = std::min(h, r.y + r.height);

    // Если пересечения нет — возвращаем пустой rect.
    if (x2 <= x1 || y2 <= y1)
        return cv::Rect(0, 0, 0, 0);

    // Возвращаем валидный прямоугольник внутри кадра.
    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

void OverlayRenderer::draw_rect_alpha(
        cv::Mat& frame,
        const cv::Rect& r,
        const cv::Scalar& color,
        float alpha
) {
    // Полностью прозрачные элементы не рисуем.
    if (alpha <= 0.0f)
        return;

    // Ограничиваем прозрачность диапазоном [0..1].
    alpha = std::min(1.0f, alpha);

    // Клиппинг по границам кадра.
    cv::Rect rr = clip_rect(r, frame.cols, frame.rows);
    if (rr.width <= 0 || rr.height <= 0)
        return;

    // Берём ROI (кусок) кадра и накладываем цвет с альфой.
    cv::Mat roi = frame(rr);
    cv::Mat overlay(roi.size(), roi.type(), color);
    cv::addWeighted(overlay, alpha, roi, 1.0f - alpha, 0.0, roi);
}

void OverlayRenderer::draw_label(
        cv::Mat& frame,
        const cv::Point& org,
        const std::string& text,
        const cv::Scalar& color
) {
    // Расчёт размеров текста для фона.
    int baseline = 0;
    cv::Size ts = cv::getTextSize(
            text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

    // Прямоугольник подложки текста.
    cv::Rect bg(
            org.x,
            org.y - ts.height - baseline - 4,
            ts.width + 6,
            ts.height + baseline + 6
    );

    // Клиппинг подложки и полупрозрачная заливка.
    bg = clip_rect(bg, frame.cols, frame.rows);
    if (bg.width > 0 && bg.height > 0)
        draw_rect_alpha(frame, bg, cv::Scalar(0, 0, 0), 0.35f);

    // Рисуем текст поверх.
    cv::putText(
            frame, text, org,
            cv::FONT_HERSHEY_SIMPLEX, 0.5,
            color, 1, cv::LINE_AA
    );
}

//------------------------------------------------------------------------------
// Рендер динамических bbox (цели трекера)
//------------------------------------------------------------------------------
void OverlayRenderer::render(
        cv::Mat& frame,
        const std::vector<Target>& targets,
        int /*selected_id*/
) const {
    const int targeting_cross = cfg_.TARGETING_CROSS_SIZE;
    // Полупрозрачная полоска HUD (верх кадра), если задана.
    if (cfg_.HUD_ALPHA > 0.0f) {
        cv::Rect hud(0, 0, frame.cols, 24);
        draw_rect_alpha(frame, hud, cv::Scalar(0, 0, 0), cfg_.HUD_ALPHA);
    }

    {
        const std::string text = "Tracks: " + std::to_string(targets.size());
        int baseline = 0;
        const cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        const int padding = 20;
        cv::Point org(frame.cols - padding - ts.width, padding + ts.height);
        draw_label(frame, org, text, cv::Scalar(255, 255, 255));
    }

    // Набор id, которые встретились в этом кадре — нужен для очистки истории.
    std::unordered_set<int> seen_ids;
    seen_ids.reserve(targets.size());

    for (const auto& t : targets) {
        seen_ids.insert(t.id);

        // ВАЖНО:
        // t.bbox — «сырой» bbox от трекера.
        // Для отображения используем сглаженный bbox.
        const cv::Rect r = smooth_bbox_for_render(t.id, t.bbox, cfg_.DYNAMIC_BBOX_WINDOW);

        // Рисуем динамический bbox:
        // - зелёный при стабильном трекинге
        // - серый при потере (missed_frames > 0)
        if (t.missed_frames > 0) {
            cv::rectangle(frame, r, cv::Scalar(128, 128, 128), 1);
        } else {
            cv::rectangle(frame, r, cv::Scalar(0, 255, 0), 1);
        }

        if (t.has_cross) {
            cv::Point center(static_cast<int>(std::round(t.cross_center.x)),
                             static_cast<int>(std::round(t.cross_center.y)));
            draw_cross(frame, center, targeting_cross, cv::Scalar(0, 0, 255), 1);
        }

#ifdef SHOW_DYNAMIC_IDS
        // Опционально подписываем id цели (если включён define).
        char buf[64];
        std::snprintf(buf, sizeof(buf), "dyn id=%d", t.id);
        draw_label(frame, cv::Point(r.x + 2, std::max(14, r.y - 2)),
                   buf, cv::Scalar(0, 0, 255));
#endif
    }

    // Очистка истории сглаживания для исчезнувших целей.
    // Если id не был замечен в этом кадре — удаляем его историю.
    for (auto it = g_dyn_smooth.begin(); it != g_dyn_smooth.end(); ) {
        if (seen_ids.find(it->first) == seen_ids.end())
            it = g_dyn_smooth.erase(it);
        else
            ++it;
    }
}

//------------------------------------------------------------------------------
// Рендер статических (ручных) bbox
//------------------------------------------------------------------------------
void OverlayRenderer::render_static_boxes(
        cv::Mat& frame,
        const std::vector<static_box>& boxes
) {
    for (const auto& sb : boxes) {
        const cv::Rect r = sb.rect;

        // Выбираем цвет в зависимости от состояния.
        cv::Scalar color;
        switch (sb.state) {
            case static_box_state::attached:
                // Статический bbox успешно привязан к цели.
                color = cv::Scalar(0, 0, 255);      // red
                break;
            case static_box_state::pending_rebind:
                // Ожидает перепривязки.
                color = cv::Scalar(0, 255, 255);    // yellow
                break;
            case static_box_state::lost:
            default:
                // Поерянный или неизвестный статус.
                color = cv::Scalar(128, 128, 128);  // gray
                break;
        }

        // Рисуем статический bbox (толще, чем динамческий).
        cv::rectangle(frame, r, color, 3);

#ifdef SHOW_IDS
        // Опционально подписываем id статического бокса.
        char buf[64];
        std::snprintf(buf, sizeof(buf), "static id=%d", sb.id);
        draw_label(frame, cv::Point(r.x + 2, std::max(14, r.y - 2)),
                   buf, color);
#endif
    }
}

//------------------------------------------------------------------------------
// Рендер статических целей (ПКМ-детекция)
//------------------------------------------------------------------------------
void OverlayRenderer::render_static_targets(
        cv::Mat& frame,
        const std::vector<StaticTarget>& targets
) const {
    for (const auto& target : targets) {
        const cv::Rect r = target.bbox;
        cv::Scalar color(255, 0, 0); // blue
        cv::rectangle(frame, r, color, 2);

#ifdef SHOW_IDS
        char buf[64];
        std::snprintf(buf, sizeof(buf), "static id=%d", target.id);
        draw_label(frame, cv::Point(r.x + 2, std::max(14, r.y - 2)), buf, color);
#endif
    }
}

bool OverlayRenderer::load_overlay_config(const toml::table &tbl) {
    try {
// ---------------------------- [overlay] ---------------------------
        const auto *overlay = tbl["overlay"].as_table();
        if (!overlay) {
            throw std::runtime_error("missing [overlay] table");
        }
        cfg_.HUD_ALPHA = read_required<float>(*overlay, "HUD_ALPHA");
        cfg_.TARGETING_CROSS_SIZE = read_required<int>(*overlay, "TARGETING_CROSS_SIZE");

// -------------------------- [smoothing] ---------------------------
        const auto *smoothing = tbl["smoothing"].as_table();
        if (!smoothing) {
            throw std::runtime_error("missing [smoothing] table");
        }
        cfg_.DYNAMIC_BBOX_WINDOW = read_required<int>(*smoothing, "DYNAMIC_BBOX_WINDOW");
        return true;

    } catch (const std::exception &e) {
        std::cerr << "overlay config load failed  " << e.what() << std::endl;
        return false;
    }
};
