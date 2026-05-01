from pathlib import Path
import tempfile
import zipfile
import json
import base64

import gradio as gr
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

from model_loader import load_pipeline
from infer import (
    default_runtime_config,
    analyze_one_image,
    aggregate_study_results,
    extract_study_archive,
    find_images,
    get_study_id_from_archive,
    strip_runtime_arrays,
    json_safe,
)


BUNDLE_PATH = Path("weights/VTSC_unified_bundle.pt")

if not BUNDLE_PATH.exists():
    raise FileNotFoundError(
        f"Не найден файл весов: {BUNDLE_PATH}. "
        "Скачайте VTSC_unified_bundle.pt и положите его в weights/"
    )

CTX = load_pipeline(BUNDLE_PATH)


CLASS_COLORS = {
    "humeral head": (255, 165, 0),
    "humerus": (0, 180, 255),
    "greater tubercle": (0, 255, 0),
    "pin": (255, 0, 255),
    "plate": (255, 255, 0),
    "fragment_humerus": (255, 0, 0),
    "fragment_tubercle": (160, 32, 240),
}

MEDICAL_CLASS_NAMES = {
    "humeral head": "Головка плечевой кости",
    "humerus": "Плечевая кость",
    "greater tubercle": "Большой бугорок плечевой кости",
    "pin": "Металлическая спица / штифт",
    "plate": "Металлическая пластина",
    "fragment_humerus": "Костный фрагмент плечевой кости",
    "fragment_tubercle": "Костный фрагмент большого бугорка",
}

CLASS_ORDER = [
    "humeral head",
    "humerus",
    "greater tubercle",
    "pin",
    "plate",
    "fragment_humerus",
    "fragment_tubercle",
]

DEFAULT_COLOR = (255, 255, 255)


HIDDEN_REVIEW_PATTERNS = [
    "ROI-детектор не нашел ROI",
    "Классификация большого бугорка выполнена по bbox",
    "Шеечно-диафизарный угол рассчитан по fallback",
    "Угол рассчитан по fallback",
    "fallback-crop из детектора костей",
    "bone_detector_fallback",
    "greater tubercle из детектора костей",
]


def class_display_name(label):
    return MEDICAL_CLASS_NAMES.get(label, label)


def rgb_to_css(rgb):
    return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"


def make_legend_html():
    items = []

    for label in CLASS_ORDER:
        color = CLASS_COLORS[label]
        medical_name = class_display_name(label)

        items.append(
            f"""
            <div class="legend-item">
                <span class="legend-dot" style="background:{rgb_to_css(color)}"></span>
                <span>{medical_name}</span>
            </div>
            """
        )

    return f"""
    <div class="legend-box">
        <b>Легенда классов детекции и сегментации:</b>
        <div class="legend-grid">
            {''.join(items)}
        </div>
    </div>
    """


def draw_label_box(draw, xy, text, color):
    x, y = xy

    try:
        bbox = draw.textbbox((x, y), text)
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2 + 6, y2 + 6], fill=color)
        draw.text((x + 3, y + 3), text, fill=(0, 0, 0))
    except Exception:
        draw.text((x, y), text, fill=color)


def draw_detections_on_image(image_pil, detections):
    img = image_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        label = detection["label_name"]
        score = detection["score"]

        color = CLASS_COLORS.get(label, DEFAULT_COLOR)
        pretty_label = class_display_name(label)

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=color,
            width=4
        )

        text = f"{pretty_label} {score:.2f}"
        draw_label_box(
            draw,
            (x1, max(0, y1 - 20)),
            text,
            color
        )

    return img


def overlay_colored_instance_masks(image_pil, seg_instances, alpha=0.45):
    img = np.array(image_pil.convert("RGB")).astype(np.float32)
    overlay = img.copy()

    for instance in seg_instances:
        mask = instance.get("_mask", None)
        label = instance.get("label_name", "unknown")

        if mask is None:
            continue

        mask_bool = mask.astype(bool)
        color = np.array(CLASS_COLORS.get(label, DEFAULT_COLOR), dtype=np.float32)

        overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * color

    return np.clip(overlay, 0, 255).astype(np.uint8)


def make_detection_segmentation_visual(result, alpha=0.45):
    image = Image.open(result["image_path"]).convert("RGB")

    seg_instances = result.get("_seg_instances", [])
    detections = result["detection"]["detections"]

    if len(seg_instances) > 0:
        overlay = overlay_colored_instance_masks(
            image,
            seg_instances,
            alpha=alpha
        )
        overlay_pil = Image.fromarray(overlay)
    else:
        overlay_pil = image.copy()

    final_img = draw_detections_on_image(
        overlay_pil,
        detections
    )

    return final_img


def is_hidden_review_reason(reason):
    if reason is None:
        return False

    text = str(reason)

    return any(pattern in text for pattern in HIDDEN_REVIEW_PATTERNS)


def sanitize_image_results_for_display(image_results):
    sanitized = []

    for result in image_results:
        item = dict(result)

        if "final" in item:
            final = dict(item["final"])

            reasons = final.get("review_reasons", [])
            visible_reasons = [
                reason for reason in reasons
                if not is_hidden_review_reason(reason)
            ]

            final["review_reasons"] = visible_reasons

            needs_review = int(len(visible_reasons) > 0)
            final["needs_expert_review"] = needs_review

            fracture = int(final.get("fracture", 0))
            foreign_body = int(final.get("foreign_body", 0))

            final["alarm"] = int(bool(fracture or foreign_body or needs_review))

            item["final"] = final

        sanitized.append(item)

    return sanitized


def sanitize_summary_for_display(summary):
    summary = json_safe(summary)

    visible_reasons = []

    for item in summary.get("final", {}).get("review_reasons", []):
        reason = item.get("reason", "")

        if not is_hidden_review_reason(reason):
            visible_reasons.append(item)

    summary["final"]["review_reasons"] = visible_reasons
    summary["final"]["needs_expert_review"] = int(len(visible_reasons) > 0)

    fracture_any = int(summary.get("fracture", {}).get("any", 0))
    foreign_body_any = int(summary.get("foreign_body", {}).get("confirmed_any", 0))
    needs_review = int(summary["final"]["needs_expert_review"])

    summary["final"]["alarm"] = int(bool(fracture_any or foreign_body_any or needs_review))

    return summary


def status_present_absent(value):
    return "присутствует" if int(value) == 1 else "отсутствует"


def status_required(value):
    return "требуется" if int(value) == 1 else "не требуется"


def status_evaluated_label(value):
    if pd.isna(value):
        return "не оценено"
    return "присутствует" if int(value) == 1 else "отсутствует"


def collect_foreign_body_types_from_results(image_results):
    found = set()

    for result in image_results or []:
        if "detection" not in result:
            continue

        for det in result["detection"].get("detections", []):
            label = det.get("label_name")

            if label in {"pin", "plate"}:
                found.add(class_display_name(label))

    return sorted(found)


def build_human_summary(summary, df, image_results=None, angle_min=125.0, angle_max=145.0):
    summary = sanitize_summary_for_display(summary)

    fracture_any = int(summary["fracture"]["any"])
    main_bone_any = int(summary["fracture"]["main_bone_any"])
    tubercle_any = int(summary["fracture"]["tubercle_any"])
    fragment_any = int(summary["fracture"]["detector_fragment_evidence_any"])

    foreign_body_any = int(summary["foreign_body"]["confirmed_any"])
    needs_review = int(summary["final"]["needs_expert_review"])

    angle_median = summary["nsa_angle"]["median"]
    foreign_body_types = collect_foreign_body_types_from_results(image_results)

    lines = []

    lines.append("## Итоговое заключение по исследованию")

    if fracture_any or foreign_body_any or needs_review:
        lines.append("**По результатам автоматического анализа выявлены признаки, требующие внимания специалиста.**")
    else:
        lines.append("**По результатам автоматического анализа критических признаков не выявлено.**")

    lines.append("")
    lines.append("### Основные результаты")

    lines.append(f"- Перелом в целом: {status_present_absent(fracture_any)}")
    lines.append(f"  - Перелом плечевой кости: {status_present_absent(main_bone_any)}")
    lines.append(f"  - Перелом большого бугорка плечевой кости: {status_present_absent(tubercle_any)}")
    lines.append(f"  - Костные фрагменты по детектору: {status_present_absent(fragment_any)}")

    lines.append("")
    lines.append(f"- Инородное тело / металлоконструкция: {status_present_absent(foreign_body_any)}")

    if foreign_body_types:
        lines.append(f"  - Обнаруженные типы по детектору: {', '.join(foreign_body_types)}")
    else:
        lines.append("  - По детектору металлические элементы не локализованы.")

    lines.append("")
    lines.append(f"- Экспертная проверка: {status_required(needs_review)}")

    lines.append("")
    lines.append("### Шеечно-диафизарный угол")

    if angle_median is None:
        lines.append("- Шеечно-диафизарный угол: не рассчитан.")
    else:
        lines.append(f"- Медианное значение по исследованию: {angle_median:.2f}°.")

        if angle_median < angle_min:
            lines.append(
                f"- Значение ниже заданного условного диапазона "
                f"({angle_min:.1f}–{angle_max:.1f}°). Рекомендуется экспертная оценка."
            )
        elif angle_median > angle_max:
            lines.append(
                f"- Значение выше заданного условного диапазона "
                f"({angle_min:.1f}–{angle_max:.1f}°). Рекомендуется экспертная оценка."
            )
        else:
            lines.append(
                f"- Значение находится в пределах заданного условного диапазона "
                f"({angle_min:.1f}–{angle_max:.1f}°)."
            )

    lines.append("")
    lines.append("### Детализация по снимкам")

    if len(df) > 0:
        for _, row in df.iterrows():
            image_name = row.get("image", "изображение")

            main_label = row.get("main_bone_fracture_label")
            tub_label = row.get("tubercle_fracture_label")
            fb_label = row.get("foreign_body_confirmed")
            angle = row.get("nsa_angle")

            main_text = status_evaluated_label(main_label)
            tub_text = status_evaluated_label(tub_label)
            fb_text = status_evaluated_label(fb_label)
            angle_text = "не рассчитан" if pd.isna(angle) else f"{float(angle):.2f}°"

            lines.append(
                f"- `{image_name}`: "
                f"перелом плечевой кости — {main_text}, "
                f"перелом большого бугорка — {tub_text}, "
                f"инородное тело — {fb_text}, "
                f"угол — {angle_text}."
            )

    visible_reasons = summary.get("final", {}).get("review_reasons", [])

    if visible_reasons:
        lines.append("")
        lines.append("### Причины экспертной проверки")

        for item in visible_reasons:
            lines.append(f"- `{item['image']}`: {item['reason']}")

    lines.append("")
    lines.append("> Автоматический вывод не является медицинским диагнозом и должен использоваться как вспомогательный результат для специалиста.")

    return "\n".join(lines)


def strip_runtime_arrays_for_report(image_results):
    clean = []

    for result in image_results:
        item = dict(result)
        item.pop("_seg_mask", None)
        item.pop("_seg_instances", None)
        clean.append(json_safe(item))

    return clean


def image_to_base64(img_path):
    with open(img_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")
    return encoded


def create_html_report(report_dir, study_id, human_summary, df, summary, visual_paths):
    report_dir = Path(report_dir)
    html_path = report_dir / "report.html"

    rows_html = df.to_html(index=False, escape=False)

    visual_blocks = []

    for visual_path in visual_paths:
        visual_path = Path(visual_path)
        encoded = image_to_base64(visual_path)

        visual_blocks.append(
            f"""
            <div class="image-card">
                <h3>{visual_path.name}</h3>
                <img src="data:image/png;base64,{encoded}">
            </div>
            """
        )

    html = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>VTSC report {study_id}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #f8fafc;
                color: #111827;
                padding: 30px;
            }}
            h1, h2, h3 {{
                color: #0f172a;
            }}
            .card {{
                background: #ffffff;
                padding: 20px;
                border-radius: 16px;
                margin-bottom: 24px;
                box-shadow: 0 8px 24px rgba(15,23,42,0.10);
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                font-size: 13px;
                background: #ffffff;
                color: #111827;
            }}
            th, td {{
                border: 1px solid #d1d5db;
                padding: 6px;
            }}
            th {{
                background: #e5e7eb;
            }}
            img {{
                max-width: 100%;
                border-radius: 12px;
                border: 1px solid #d1d5db;
            }}
            .image-card {{
                background: #ffffff;
                padding: 16px;
                border-radius: 16px;
                margin-bottom: 24px;
                box-shadow: 0 8px 24px rgba(15,23,42,0.10);
            }}
            pre {{
                white-space: pre-wrap;
                font-size: 15px;
                line-height: 1.5;
            }}
        </style>
    </head>
    <body>
        <h1>VTSC: отчёт по исследованию {study_id}</h1>

        <div class="card">
            <h2>Итоговое описание</h2>
            <pre>{human_summary}</pre>
        </div>

        <div class="card">
            <h2>Таблица по снимкам</h2>
            {rows_html}
        </div>

        <div class="card">
            <h2>JSON summary</h2>
            <pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
        </div>

        <h2>Визуализации</h2>
        {''.join(visual_blocks)}
    </body>
    </html>
    """

    with open(html_path, "w", encoding="utf-8") as file:
        file.write(html)

    return html_path


def create_report_zip(report_dir, study_id):
    report_dir = Path(report_dir)
    zip_path = report_dir.parent / f"{study_id}_VTSC_report.zip"

    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for path in report_dir.rglob("*"):
            if path.is_file():
                zip_file.write(path, path.relative_to(report_dir))

    return str(zip_path)


def make_display_table(df):
    df_display = df.copy()

    rename_cols = {
        "image": "Снимок",
        "projection": "Проекция",
        "foreign_body_confirmed": "Инородное тело",
        "main_bone_fracture_label": "Перелом плечевой кости",
        "tubercle_fracture_label": "Перелом большого бугорка",
        "nsa_angle": "Шеечно-диафизарный угол",
        "needs_expert_review": "Экспертная проверка",
        "alarm": "Тревожный флаг",
    }

    df_display = df_display.rename(columns=rename_cols)

    for col in [
        "Инородное тело",
        "Перелом плечевой кости",
        "Перелом большого бугорка",
        "Экспертная проверка",
        "Тревожный флаг",
    ]:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: "присутствует" if pd.notna(x) and int(x) == 1
                else "отсутствует" if pd.notna(x)
                else "не оценено"
            )

    return df_display


def analyze_archive_for_gradio(
    archive_file,
    projection_conf_thr,
    foreign_body_thr,
    main_fracture_thr,
    tubercle_thr,
    detector_score_thr,
    roi_score_thr,
    seg_alpha,
    angle_min,
    angle_max,
    max_images_to_show
):
    if archive_file is None:
        return (
            "Загрузите архив `.zip` с исследованием.",
            pd.DataFrame(),
            [],
            None,
            {}
        )

    archive_path = Path(archive_file)

    runtime_config = default_runtime_config()
    runtime_config["projection_conf_thr"] = float(projection_conf_thr)
    runtime_config["foreign_body_thr"] = float(foreign_body_thr)
    runtime_config["main_fracture_thr"] = float(main_fracture_thr)
    runtime_config["tubercle_thr"] = float(tubercle_thr)
    runtime_config["detector_score_thr"] = float(detector_score_thr)
    runtime_config["roi_score_thr"] = float(roi_score_thr)

    study_id = get_study_id_from_archive(archive_path)

    work_root = Path(tempfile.mkdtemp(prefix=f"vtsc_{study_id}_"))
    extract_root = work_root / "input"
    report_dir = work_root / "report"
    visual_dir = report_dir / "visuals"

    extract_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    extract_study_archive(archive_path, extract_root)

    image_paths = find_images(extract_root)

    if len(image_paths) == 0:
        return (
            "В архиве не найдено изображений.",
            pd.DataFrame(),
            [],
            None,
            {}
        )

    image_results = []

    for image_path in image_paths:
        rel_path = image_path.relative_to(extract_root)

        try:
            result = analyze_one_image(
                CTX,
                image_path,
                rel_path=rel_path,
                runtime_config=runtime_config,
                keep_seg_mask=True
            )
        except Exception as exc:
            result = {
                "image_path": str(image_path),
                "relative_path": str(rel_path),
                "error": str(exc),
                "final": {
                    "alarm": 1,
                    "needs_expert_review": 1,
                    "review_reasons": [
                        f"Ошибка обработки изображения: {exc}"
                    ]
                }
            }

        image_results.append(result)

    image_results = sanitize_image_results_for_display(image_results)

    gallery_items = []
    visual_paths = []

    for index, result in enumerate(image_results, 1):
        if "error" in result:
            continue

        vis_img = make_detection_segmentation_visual(
            result,
            alpha=float(seg_alpha)
        )

        caption = (
            f"{result['relative_path']} | "
            f"проекция={result['projection']['label']} | "
            f"перелом={status_present_absent(result['final']['fracture'])} | "
            f"инородное тело={status_present_absent(result['final']['foreign_body'])} | "
            f"экспертная проверка={status_required(result['final']['needs_expert_review'])}"
        )

        visual_path = visual_dir / f"visual_{index:03d}.png"
        vis_img.save(visual_path)

        visual_paths.append(visual_path)

        if max_images_to_show == -1 or len(gallery_items) < int(max_images_to_show):
            gallery_items.append((vis_img, caption))

    clean_results = strip_runtime_arrays_for_report(image_results)

    df, summary = aggregate_study_results(
        clean_results,
        study_id=study_id
    )

    summary_for_display = sanitize_summary_for_display(summary)

    human_summary = build_human_summary(
        summary_for_display,
        df,
        image_results=clean_results,
        angle_min=float(angle_min),
        angle_max=float(angle_max)
    )

    df_display = make_display_table(df)

    csv_path = report_dir / "image_level_results.csv"
    json_summary_path = report_dir / "study_summary.json"
    json_full_path = report_dir / "full_image_results.json"
    txt_path = report_dir / "human_summary.txt"

    df_display.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with open(json_summary_path, "w", encoding="utf-8") as file:
        json.dump(json_safe(summary_for_display), file, ensure_ascii=False, indent=2)

    with open(json_full_path, "w", encoding="utf-8") as file:
        json.dump(json_safe(clean_results), file, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(human_summary)

    create_html_report(
        report_dir=report_dir,
        study_id=study_id,
        human_summary=human_summary,
        df=df_display,
        summary=summary_for_display,
        visual_paths=visual_paths
    )

    zip_report = create_report_zip(
        report_dir=report_dir,
        study_id=study_id
    )

    return (
        human_summary,
        df_display,
        gallery_items,
        zip_report,
        summary_for_display
    )


CUSTOM_CSS = """
.gradio-container {
    max-width: 1400px !important;
    background: #f8fafc !important;
}

.main-title {
    text-align: center;
    padding: 26px 18px;
    border-radius: 22px;
    background: linear-gradient(135deg, #0f172a, #1e293b) !important;
    color: #ffffff !important;
    margin: 18px 0 24px 0;
    box-shadow: 0 12px 32px rgba(15, 23, 42, 0.25);
}

.main-title h1 {
    color: #ffffff !important;
    font-size: 32px !important;
    font-weight: 800 !important;
    margin: 0 0 10px 0 !important;
    letter-spacing: 0.2px;
}

.main-title p {
    color: #cbd5e1 !important;
    font-size: 17px !important;
    margin: 0 !important;
}

.legend-box {
    background: #ffffff;
    color: #111827;
    border-radius: 16px;
    padding: 14px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.08);
}

.legend-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 10px;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 7px;
    background: #f1f5f9;
    color: #111827;
    padding: 7px 11px;
    border-radius: 999px;
    font-size: 14px;
}

.legend-dot {
    width: 14px;
    height: 14px;
    display: inline-block;
    border-radius: 50%;
    border: 1px solid rgba(0,0,0,0.25);
}
"""


with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML(
        """
        <div class="main-title">
            <h1>VTSC: автоматизированный анализ рентгенограмм плечевой области</h1>
            <p>Загрузка исследования архивом → детекция, сегментация, классификация, расчет угла и формирование отчета</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            archive_input = gr.File(
                label="Загрузите архив исследования (.zip)",
                file_types=[".zip"],
                type="filepath"
            )

            gr.HTML(make_legend_html())

            with gr.Accordion("Настройки порогов", open=True):
                projection_conf_thr = gr.Slider(
                    minimum=0.50,
                    maximum=0.99,
                    value=0.75,
                    step=0.01,
                    label="Порог уверенности классификатора проекции"
                )

                foreign_body_thr = gr.Slider(
                    minimum=0.10,
                    maximum=0.99,
                    value=0.80,
                    step=0.01,
                    label="Порог классификатора инородного тела"
                )

                main_fracture_thr = gr.Slider(
                    minimum=0.10,
                    maximum=0.99,
                    value=0.50,
                    step=0.01,
                    label="Порог классификатора перелома плечевой кости"
                )

                tubercle_thr = gr.Slider(
                    minimum=0.10,
                    maximum=0.99,
                    value=0.50,
                    step=0.01,
                    label="Порог классификатора перелома большого бугорка"
                )

                detector_score_thr = gr.Slider(
                    minimum=0.01,
                    maximum=0.95,
                    value=0.30,
                    step=0.01,
                    label="Порог уверенности детектора костей"
                )

                roi_score_thr = gr.Slider(
                    minimum=0.01,
                    maximum=0.95,
                    value=0.30,
                    step=0.01,
                    label="Порог уверенности ROI"
                )

                seg_alpha = gr.Slider(
                    minimum=0.10,
                    maximum=0.90,
                    value=0.45,
                    step=0.05,
                    label="Прозрачность цветной сегментации"
                )

            with gr.Accordion("Интерпретация шеечно-диафизарного угла", open=False):
                angle_min = gr.Number(
                    value=125.0,
                    label="Нижняя граница условного диапазона угла"
                )

                angle_max = gr.Number(
                    value=145.0,
                    label="Верхняя граница условного диапазона угла"
                )

            max_images_to_show = gr.Radio(
                choices=[
                    ("Показать 5 изображений", 5),
                    ("Показать 10 изображений", 10),
                    ("Показать все изображения", -1),
                ],
                value=5,
                label="Количество визуализаций"
            )

            run_btn = gr.Button(
                "Запустить анализ",
                variant="primary"
            )

        with gr.Column(scale=2):
            human_output = gr.Markdown(
                label="Итоговое описание"
            )

            report_file = gr.File(
                label="Скачать отчет"
            )

            gallery = gr.Gallery(
                label="Цветная детекция + сегментация",
                columns=2,
                height=650,
                object_fit="contain"
            )

            results_table = gr.Dataframe(
                label="Таблица результатов по снимкам",
                wrap=True
            )

            json_output = gr.JSON(
                label="JSON summary"
            )

    run_btn.click(
        fn=analyze_archive_for_gradio,
        inputs=[
            archive_input,
            projection_conf_thr,
            foreign_body_thr,
            main_fracture_thr,
            tubercle_thr,
            detector_score_thr,
            roi_score_thr,
            seg_alpha,
            angle_min,
            angle_max,
            max_images_to_show,
        ],
        outputs=[
            human_output,
            results_table,
            gallery,
            report_file,
            json_output,
        ]
    )


if __name__ == "__main__":
    demo.launch()
