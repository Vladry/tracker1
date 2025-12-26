#include "overlay.h"
//#include <algorithm>
#include <deque>
#include <unordered_map>
#include <unordered_set>

//==============================================================================
// Визуальное сглаживание динамических bbox (рамок)
//
// ВАЖНО:
//  - Это влияет ТОЛЬКО на отображение (рендер).
//  - Логика трекера/детектора не меняется.
//  - Для каждого динамического Target.id хранится своя история.
//==============================================================================

// Количество последних кадров, используемых для сглаживания.
// Чем больше окно, тем плавнее, но и больше визуальная задержка.
static constexpr int DYN_SMOOTH_WINDOW = 15;

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
static cv::Rect smooth_bbox_for_render(int id, const cv::Rect& current) {
    // Достаём (или создаём) историю для данного id.
    auto& st = g_dyn_smooth[id];

    // Кладём текущий bbox в историю (в float-формате для точности).
    st.history.emplace_back(cv::Rect2f(current));

    // Поддерживаем ограничение на длину истории.
    if ((int)st.history.size() > DYN_SMOOTH_WINDOW)
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
    // Полупрозрачная полоска HUD (верх кадра), если задана.
    if (cfg_.hud_alpha > 0.0f) {
        cv::Rect hud(0, 0, frame.cols, 24);
        draw_rect_alpha(frame, hud, cv::Scalar(0, 0, 0), cfg_.hud_alpha);
    }

    // Набор id, которые встретились в этом кадре — нужен для очистки истории.
    std::unordered_set<int> seen_ids;
    seen_ids.reserve(targets.size());

    for (const auto& t : targets) {
        seen_ids.insert(t.id);

        // ВАЖНО:
        // t.bbox — «сырой» bbox от трекера.
        // Для отображения используем сглаженный bbox.
        const cv::Rect r = smooth_bbox_for_render(t.id, t.bbox);

        // Рисуем динамический bbox (зелёный, тонкая линия).
        cv::rectangle(frame, r, cv::Scalar(0, 255, 0), 1);

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
                // Потерянный или неизвестный статус.
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

bool OverlayRenderer::load_overlay_config(const toml::table &tbl) {
    try {
// ---------------------------- [overlay] ---------------------------
        const auto *overlay = tbl["overlay"].as_table();
        if (!overlay) {
            throw std::runtime_error("missing [overlay] table");
        }
        cfg_.hud_alpha = read_required<float>(*overlay, "hud_alpha");
        cfg_.unselected_alpha_when_selected = read_required<float>(
                *overlay, "unselected_alpha_when_selected");

// -------------------------- [smoothing] ---------------------------
        const auto *smoothing = tbl["smoothing"].as_table();
        if (!smoothing) {
            throw std::runtime_error("missing [smoothing] table");
        }
        cfg_.dynamic_bbox_window = read_required<int>(*smoothing, "dynamic_bbox_window");
        return true;

    } catch (const std::exception &e) {
        std::cerr << "overlay config load failed  " << e.what() << std::endl;
        return false;
    }
};
