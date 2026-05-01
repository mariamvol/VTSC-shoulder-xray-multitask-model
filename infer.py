import math
import zipfile
import shutil
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
import torch.nn.functional as F
from torchvision import transforms as T
import torchvision.transforms.functional as TF

from model_loader import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    NSA_IMG_SIZE,
    ROI_DETECTOR_IMG_MAX,
)


DETECTOR_LABELS = {
    1: "humeral head",
    2: "humerus",
    3: "greater tubercle",
    4: "pin",
    5: "plate",
    6: "fragment_humerus",
    7: "fragment_tubercle",
}

HUMERUS_CLASS_NAMES = {"humerus"}

FOREIGN_BODY_DETECTOR_CLASS_NAMES = {
    "pin",
    "plate",
}

FRACTURE_EVIDENCE_CLASS_NAMES = {
    "fragment_humerus",
    "fragment_tubercle",
}

TUBERCLE_FALLBACK_CLASS_NAMES = {
    "greater tubercle",
}

ANGLE_FALLBACK_CLASS_NAMES = {
    "humeral head",
    "humerus",
    "greater tubercle",
}


def default_runtime_config():
    return {
        "projection_conf_thr": 0.75,
        "foreign_body_thr": 0.80,
        "main_fracture_thr": 0.50,
        "tubercle_thr": 0.50,
        "detector_score_thr": 0.30,
        "roi_score_thr": 0.30,
    }


class SquarePad:
    def __call__(self, image):
        width, height = image.size
        max_side = max(width, height)

        left = (max_side - width) // 2
        top = (max_side - height) // 2
        right = max_side - width - left
        bottom = max_side - height - top

        return ImageOps.expand(image, (left, top, right, bottom), fill=0)


def make_cls_transform(img_size, mean=IMAGENET_MEAN, std=IMAGENET_STD, square_pad=False):
    steps = []

    if square_pad:
        steps.append(SquarePad())

    steps.extend([
        T.Resize((int(img_size), int(img_size))),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    return T.Compose(steps)


def pil_rgb(path):
    return Image.open(path).convert("RGB")


def make_detector_tensor(image_pil):
    arr = np.array(image_pil.convert("RGB")).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor


@torch.no_grad()
def predict_projection(ctx, image_pil):
    models = ctx["models"]
    metas = ctx["metas"]
    device = ctx["device"]

    meta = metas["projection"]

    transform = make_cls_transform(
        meta["img_size"],
        meta["mean"],
        meta["std"],
        square_pad=False,
    )

    x = transform(image_pil).unsqueeze(0).to(device)

    logits = models["projection"](x)

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    classes = meta.get("classes") or {"D": 0, "S": 1}
    idx_to_class = {int(v): k for k, v in classes.items()}

    idx = int(np.argmax(probs))
    label = idx_to_class[idx]
    conf = float(probs[idx])

    prob_dict = {
        idx_to_class[i]: float(probs[i])
        for i in range(len(probs))
    }

    return label, conf, prob_dict


@torch.no_grad()
def predict_binary_prob(ctx, model_key, image_pil):
    models = ctx["models"]
    metas = ctx["metas"]
    device = ctx["device"]

    meta = metas[model_key]

    transform = make_cls_transform(
        meta.get("img_size", 224),
        meta.get("mean", IMAGENET_MEAN),
        meta.get("std", IMAGENET_STD),
        square_pad=meta.get("square_pad", False),
    )

    x = transform(image_pil).unsqueeze(0).to(device)

    logits = models[model_key](x)

    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    logits = logits.float()

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    if logits.shape[1] == 1:
        prob = torch.sigmoid(logits)[0, 0].item()
    else:
        prob = torch.softmax(logits, dim=1)[0, 1].item()

    return float(prob)


def pred2_to_deg(y2):

    y = F.normalize(y2, dim=1)

    c = y[:, 0]
    s = y[:, 1]

    ang2 = torch.atan2(s, c)
    ang = 0.5 * ang2

    ang = torch.abs(ang)
    ang = torch.clamp(ang, 0, math.pi / 2)

    return ang * 180.0 / math.pi


def obtuse_from_acute(a_deg):
    a = float(a_deg)
    a = max(0.0, min(90.0, a))
    return 180.0 - a


@torch.no_grad()
def predict_nsa(ctx, roi_crop_pil):
    models = ctx["models"]
    device = ctx["device"]

    crop_gray = np.array(roi_crop_pil.convert("L"))

    crop_resized = cv2.resize(
        crop_gray,
        (NSA_IMG_SIZE, NSA_IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_GRAY2RGB)

    x = TF.to_tensor(crop_rgb)
    x = (x - 0.5) / 0.5
    x = x.unsqueeze(0).to(device)

    y2 = models["nsa_regressor"](x)

    acute_deg = float(pred2_to_deg(y2).item())
    obtuse_deg = obtuse_from_acute(acute_deg)

    return {
        "raw_outputs": y2.detach().cpu().view(-1).numpy().astype(float).tolist(),
        "acute_angle": float(acute_deg),
        "angle": float(obtuse_deg),
    }


@torch.no_grad()
def run_bone_detector(ctx, detector_key, image_pil):
    models = ctx["models"]
    metas = ctx["metas"]
    device = ctx["device"]

    label_map = metas[detector_key]["label_map"]

    x = make_detector_tensor(image_pil).to(device)

    pred = models[detector_key]([x])[0]

    boxes = pred["boxes"].detach().cpu().numpy()
    labels = pred["labels"].detach().cpu().numpy()
    scores = pred["scores"].detach().cpu().numpy()

    detections = []

    for box, label_id, score in zip(boxes, labels, scores):
        label_id = int(label_id)

        detections.append({
            "box": [float(v) for v in box],
            "label_id": label_id,
            "label_name": label_map.get(label_id, f"class_{label_id}"),
            "score": float(score),
        })

    return detections


def filter_detections_by_names(detections, class_names):
    return [
        detection for detection in detections
        if detection["label_name"] in class_names
    ]


def choose_best_detection(detections, class_names):
    candidates = filter_detections_by_names(detections, class_names)

    if len(candidates) == 0:
        return None

    return sorted(candidates, key=lambda x: x["score"], reverse=True)[0]


def choose_best_any_detection(detections):
    if len(detections) == 0:
        return None

    return sorted(detections, key=lambda x: x["score"], reverse=True)[0]


def has_any_detection(detections, class_names):
    return len(filter_detections_by_names(detections, class_names)) > 0


def crop_with_padding_px(image_pil, box, pad_px=10):
    width, height = image_pil.size
    x1, y1, x2, y2 = box

    x1 = max(0, int(x1 - pad_px))
    y1 = max(0, int(y1 - pad_px))
    x2 = min(width, int(x2 + pad_px))
    y2 = min(height, int(y2 + pad_px))

    if x2 <= x1 or y2 <= y1:
        return None

    return image_pil.crop((x1, y1, x2, y2))


def crop_with_padding_rel(image_pil, box, pad=0.10):
    width, height = image_pil.size
    x1, y1, x2, y2 = box

    box_w = x2 - x1
    box_h = y2 - y1

    x1 = max(0, int(x1 - box_w * pad))
    y1 = max(0, int(y1 - box_h * pad))
    x2 = min(width, int(x2 + box_w * pad))
    y2 = min(height, int(y2 + box_h * pad))

    if x2 <= x1 or y2 <= y1:
        return None

    return image_pil.crop((x1, y1, x2, y2))


def get_humerus_crop_for_main_bone(image_pil, detections):
    """
    Для основной кости:
    - ищем humerus;
    - если нет humerus, берём лучший bbox;
    - если вообще нет bbox, берём всё изображение.
    """
    humerus_det = choose_best_detection(detections, HUMERUS_CLASS_NAMES)

    fallback_used = False
    fallback_reason = None

    if humerus_det is None:
        humerus_det = choose_best_any_detection(detections)
        fallback_used = True
        fallback_reason = "best_detection_fallback"

    if humerus_det is None:
        return image_pil, None, True, "no_detection_fallback_full_image"

    crop = crop_with_padding_px(
        image_pil,
        humerus_det["box"],
        pad_px=10
    )

    if crop is None:
        return image_pil, humerus_det, True, "bad_box_fallback_full_image"

    return crop, humerus_det, fallback_used, fallback_reason


def _to_int_box_xyxy(box, width, height):
    x1, y1, x2, y2 = box

    x1 = int(round(float(x1)))
    y1 = int(round(float(y1)))
    x2 = int(round(float(x2)))
    y2 = int(round(float(y2)))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    x1 = int(max(0, min(width - 1, x1)))
    y1 = int(max(0, min(height - 1, y1)))
    x2 = int(max(0, min(width - 1, x2)))
    y2 = int(max(0, min(height - 1, y2)))

    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)

    return x1, y1, x2, y2


@torch.no_grad()
def get_roi_crop_from_roi_detector(
    ctx,
    image_pil,
    score_thr=0.30,
    det_img_max=ROI_DETECTOR_IMG_MAX,
):

    models = ctx["models"]
    device = ctx["device"]

    roi_model = models["roi_detector"]
    roi_model.eval()

    img_rgb_np = np.array(image_pil.convert("RGB"))
    img_gray = cv2.cvtColor(img_rgb_np, cv2.COLOR_RGB2GRAY)

    height0, width0 = img_gray.shape[:2]
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    scale = 1.0

    if max(height0, width0) > det_img_max:
        scale = det_img_max / max(height0, width0)
        new_width = int(round(width0 * scale))
        new_height = int(round(height0 * scale))
        img_det = cv2.resize(img_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        img_det = img_rgb

    x_det = TF.to_tensor(img_det).to(device)

    pred = roi_model([x_det])[0]

    boxes = pred["boxes"].detach().cpu()
    scores = pred["scores"].detach().cpu()
    labels = pred["labels"].detach().cpu()

    roi_detections = []

    for box_scaled, score, label_id in zip(boxes, scores, labels):
        box_scaled = box_scaled.numpy().tolist()
        box_orig = [float(v) / scale for v in box_scaled]

        x1, y1, x2, y2 = _to_int_box_xyxy(box_orig, width0, height0)

        roi_detections.append({
            "box": [float(x1), float(y1), float(x2), float(y2)],
            "label_id": int(label_id.item()),
            "label_name": "roi",
            "score": float(score.item()),
        })

    if len(roi_detections) == 0:
        return None, None, roi_detections

    best_roi = sorted(roi_detections, key=lambda d: d["score"], reverse=True)[0]
    best_roi["low_score"] = bool(best_roi["score"] < score_thr)

    x1, y1, x2, y2 = map(int, best_roi["box"])

    crop_gray = img_gray[y1:y2, x1:x2].copy()

    if crop_gray.size == 0:
        return None, best_roi, roi_detections

    crop_rgb = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2RGB)
    crop_pil = Image.fromarray(crop_rgb)

    return crop_pil, best_roi, roi_detections


def get_tubercle_fallback_crop_from_bone_detector(image_pil, detections, padding=0.12):
    tubercle_det = choose_best_detection(
        detections,
        TUBERCLE_FALLBACK_CLASS_NAMES
    )

    if tubercle_det is None:
        return None, None

    crop = crop_with_padding_rel(
        image_pil,
        tubercle_det["box"],
        pad=padding
    )

    if crop is None:
        return None, None

    info = {
        "source": "bone_detector_greater_tubercle_fallback",
        "label_name": tubercle_det["label_name"],
        "score": float(tubercle_det["score"]),
        "box": tubercle_det["box"],
        "padding": float(padding),
    }

    return crop, info


def get_angle_fallback_crop_from_bone_detector(image_pil, detections, padding=0.12):
    preferred = [
        detection for detection in detections
        if detection["label_name"] in ANGLE_FALLBACK_CLASS_NAMES
    ]

    if len(preferred) == 0:
        return None, None

    xs1 = [d["box"][0] for d in preferred]
    ys1 = [d["box"][1] for d in preferred]
    xs2 = [d["box"][2] for d in preferred]
    ys2 = [d["box"][3] for d in preferred]

    union_box = [
        float(min(xs1)),
        float(min(ys1)),
        float(max(xs2)),
        float(max(ys2)),
    ]

    crop = crop_with_padding_rel(
        image_pil,
        union_box,
        pad=padding
    )

    if crop is None:
        return None, None

    info = {
        "source": "bone_detector_fallback",
        "used_labels": sorted(list(set(d["label_name"] for d in preferred))),
        "num_boxes": int(len(preferred)),
        "union_box": union_box,
        "padding": float(padding),
    }

    return crop, info


def box_with_padding_coords(image_pil, box, pad=0.12):
    width, height = image_pil.size
    x1, y1, x2, y2 = box

    box_w = x2 - x1
    box_h = y2 - y1

    x1p = max(0, int(x1 - box_w * pad))
    y1p = max(0, int(y1 - box_h * pad))
    x2p = min(width, int(x2 + box_w * pad))
    y2p = min(height, int(y2 + box_h * pad))

    return [x1p, y1p, x2p, y2p]


@torch.no_grad()
def predict_roi_segmentation_mask(ctx, seg_key, roi_pil):
    models = ctx["models"]
    metas = ctx["metas"]
    device = ctx["device"]

    seg_meta = metas[seg_key]

    transform = make_cls_transform(
        seg_meta["img_size"],
        seg_meta["mean"],
        seg_meta["std"],
        square_pad=False,
    )

    x = transform(roi_pil).unsqueeze(0).to(device)

    logits = models[seg_key](x)

    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    mask = (prob >= seg_meta["threshold"]).astype(np.uint8)

    return mask, prob


def paste_roi_mask_to_full(full_shape_hw, roi_box, roi_mask):
    full_h, full_w = full_shape_hw
    x1, y1, x2, y2 = [int(v) for v in roi_box]

    x1 = max(0, min(x1, full_w))
    x2 = max(0, min(x2, full_w))
    y1 = max(0, min(y1, full_h))
    y2 = max(0, min(y2, full_h))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((full_h, full_w), dtype=np.uint8)

    roi_w = x2 - x1
    roi_h = y2 - y1

    resized_mask = cv2.resize(
        roi_mask.astype(np.uint8),
        (roi_w, roi_h),
        interpolation=cv2.INTER_NEAREST
    )

    full_mask = np.zeros((full_h, full_w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = resized_mask

    return full_mask


def run_roi_segmentation_from_detections(ctx, seg_key, image_pil, detections):
    metas = ctx["metas"]

    seg_meta = metas[seg_key]

    img_w, img_h = image_pil.size
    full_shape_hw = (img_h, img_w)

    full_binary_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    instance_masks = []

    for det in detections:
        padded_box = box_with_padding_coords(
            image_pil,
            det["box"],
            pad=seg_meta["roi_margin"]
        )

        roi_pil = image_pil.crop(tuple(padded_box))

        if roi_pil.size[0] <= 1 or roi_pil.size[1] <= 1:
            continue

        roi_mask, _ = predict_roi_segmentation_mask(
            ctx,
            seg_key,
            roi_pil
        )

        full_instance_mask = paste_roi_mask_to_full(
            full_shape_hw=full_shape_hw,
            roi_box=padded_box,
            roi_mask=roi_mask
        )

        full_binary_mask = np.maximum(full_binary_mask, full_instance_mask)

        instance_masks.append({
            "label_id": int(det["label_id"]),
            "label_name": det["label_name"],
            "score": float(det["score"]),
            "detector_box": det["box"],
            "seg_roi_box": padded_box,
            "mask_area": int(full_instance_mask.sum()),
            "_mask": full_instance_mask,
        })

    return full_binary_mask, instance_masks


def check_foreign_body_consistency(detections, fb_classifier_prob, fb_threshold=0.8):
    fb_by_detector = has_any_detection(
        detections,
        FOREIGN_BODY_DETECTOR_CLASS_NAMES
    )

    fb_by_classifier = fb_classifier_prob >= fb_threshold

    needs_expert_review = False
    reason = None

    if fb_by_detector and fb_by_classifier:
        status = "confirmed_present"

    elif (not fb_by_detector) and (not fb_by_classifier):
        status = "confirmed_absent"

    elif fb_by_detector and (not fb_by_classifier):
        status = "conflict_detector_yes_classifier_no"
        needs_expert_review = True
        reason = (
            "Детектор локализовал возможное инородное тело "
            "(pin/plate), но классификатор инородного тела не подтвердил его наличие."
        )

    else:
        status = "conflict_classifier_yes_detector_no"
        needs_expert_review = True
        reason = (
            "Классификатор определил возможное наличие инородного тела, "
            "но детектор не смог локализовать pin/plate."
        )

    return {
        "fb_by_detector": int(fb_by_detector),
        "fb_by_classifier": int(fb_by_classifier),
        "status": status,
        "needs_expert_review": bool(needs_expert_review),
        "reason": reason,
    }


def get_detector_fracture_evidence(detections):
    fragments = filter_detections_by_names(
        detections,
        FRACTURE_EVIDENCE_CLASS_NAMES
    )

    return {
        "has_fragment_evidence": int(len(fragments) > 0),
        "fragments": fragments,
    }


def analyze_one_image(ctx, image_path, rel_path=None, runtime_config=None, keep_seg_mask=True):
    if runtime_config is None:
        runtime_config = default_runtime_config()

    image_path = Path(image_path)
    image = pil_rgb(image_path)

    models = ctx["models"]

    models["detector_D"].roi_heads.score_thresh = float(runtime_config["detector_score_thr"])
    models["detector_S"].roi_heads.score_thresh = float(runtime_config["detector_score_thr"])

    review_reasons = []
    needs_expert_review = False

    projection, projection_conf, projection_probs = predict_projection(ctx, image)

    if projection == "D":
        detector_key = "detector_D"
        seg_key = "seg_D"
        fb_key = "foreign_body_D"
        main_key = "main_bone_D"
    elif projection == "S":
        detector_key = "detector_S"
        seg_key = "seg_S"
        fb_key = "foreign_body_S"
        main_key = "main_bone_S"
    else:
        raise ValueError(f"Неизвестная проекция: {projection}")

    if projection_conf < float(runtime_config["projection_conf_thr"]):
        needs_expert_review = True
        review_reasons.append(
            f"Классификатор проекции не уверен: projection={projection}, confidence={projection_conf:.3f}."
        )

    detections = run_bone_detector(ctx, detector_key, image)

    seg_mask, seg_instances = run_roi_segmentation_from_detections(
        ctx,
        seg_key,
        image,
        detections
    )

    fb_prob = predict_binary_prob(ctx, fb_key, image)

    fb_check = check_foreign_body_consistency(
        detections=detections,
        fb_classifier_prob=fb_prob,
        fb_threshold=float(runtime_config["foreign_body_thr"]),
    )

    if fb_check["needs_expert_review"]:
        needs_expert_review = True
        review_reasons.append(fb_check["reason"])

    fracture_evidence = get_detector_fracture_evidence(detections)

    humerus_crop, humerus_detection, humerus_fallback_used, humerus_fallback_reason = get_humerus_crop_for_main_bone(
        image,
        detections
    )

    if humerus_fallback_used:
        needs_expert_review = True

        if humerus_fallback_reason is not None:
            review_reasons.append(
                f"Humerus не был корректно локализован. Использован fallback: {humerus_fallback_reason}."
            )
        else:
            review_reasons.append(
                "Humerus не был локализован как отдельный класс, использован лучший bbox другого класса."
            )

    main_bone_fracture_prob = predict_binary_prob(
        ctx,
        main_key,
        humerus_crop
    )

    main_bone_fracture_label = int(
        main_bone_fracture_prob >= float(runtime_config["main_fracture_thr"])
    )

    roi_crop, roi_detection, roi_detections = get_roi_crop_from_roi_detector(
        ctx,
        image,
        score_thr=float(runtime_config["roi_score_thr"]),
        det_img_max=ROI_DETECTOR_IMG_MAX,
    )

    tubercle_crop = None
    tubercle_source = None
    tubercle_fallback_info = None

    if roi_crop is not None:
        tubercle_crop = roi_crop
        tubercle_source = "roi_detector"

        if roi_detection is not None and roi_detection.get("low_score", False):
            needs_expert_review = True
            review_reasons.append(
                f"ROI большого бугорка/шейки найден с низкой уверенностью: score={roi_detection['score']:.3f}."
            )

    else:
        fallback_tubercle_crop, fallback_tubercle_info = get_tubercle_fallback_crop_from_bone_detector(
            image,
            detections,
            padding=0.12
        )

        if fallback_tubercle_crop is not None:
            tubercle_crop = fallback_tubercle_crop
            tubercle_source = "bone_detector_greater_tubercle_fallback"
            tubercle_fallback_info = fallback_tubercle_info

            needs_expert_review = True
            review_reasons.append(
                "ROI-детектор не нашёл ROI. Классификация большого бугорка выполнена по bbox greater tubercle из детектора костей."
            )

        else:
            needs_expert_review = True
            review_reasons.append(
                "ROI-детектор не нашёл ROI, и детектор костей не нашёл greater tubercle. Классификация большого бугорка не выполнена."
            )

    if tubercle_crop is not None:
        tubercle_prob = predict_binary_prob(ctx, "tubercle", tubercle_crop)
        tubercle_label = int(
            tubercle_prob >= float(runtime_config["tubercle_thr"])
        )
    else:
        tubercle_prob = None
        tubercle_label = None

    angle_crop = None
    angle_source = None
    angle_fallback_info = None

    if roi_crop is not None:
        angle_crop = roi_crop
        angle_source = "roi_detector"

    else:
        fallback_angle_crop, fallback_angle_info = get_angle_fallback_crop_from_bone_detector(
            image,
            detections,
            padding=0.12
        )

        if fallback_angle_crop is not None:
            angle_crop = fallback_angle_crop
            angle_source = "bone_detector_fallback"
            angle_fallback_info = fallback_angle_info

            needs_expert_review = True
            review_reasons.append(
                "ROI-детектор не нашёл ROI. Шеечно-диафизарный угол рассчитан по fallback-crop из детектора костей."
            )

    if angle_crop is not None:
        nsa_pred = predict_nsa(ctx, angle_crop)

        nsa_angle = nsa_pred.get("angle")
        nsa_acute_angle = nsa_pred.get("acute_angle")
        nsa_raw_outputs = nsa_pred.get("raw_outputs")

    else:
        nsa_angle = None
        nsa_acute_angle = None
        nsa_raw_outputs = None

        needs_expert_review = True
        review_reasons.append(
            "Не удалось получить crop для расчёта шеечно-диафизарного угла ни из ROI-детектора, ни из детектора костей."
        )

    main_is_zero = main_bone_fracture_label == 0
    tubercle_is_zero = tubercle_label == 0 if tubercle_label is not None else False

    if fracture_evidence["has_fragment_evidence"] == 1:
        if main_is_zero and tubercle_is_zero:
            needs_expert_review = True
            review_reasons.append(
                "Детектор обнаружил костный фрагмент, но классификаторы перелома не подтвердили перелом."
            )

    main_fracture = bool(main_bone_fracture_label == 1)
    tubercle_fracture = bool(tubercle_label == 1) if tubercle_label is not None else False

    fracture_any = bool(main_fracture or tubercle_fracture)
    foreign_body_confirmed = fb_check["status"] == "confirmed_present"

    alarm = bool(
        fracture_any
        or foreign_body_confirmed
        or needs_expert_review
    )

    result = {
        "image_path": str(image_path),
        "relative_path": str(rel_path) if rel_path is not None else image_path.name,

        "projection": {
            "label": projection,
            "confidence": float(projection_conf),
            "probs": projection_probs,
            "uncertain": bool(projection_conf < float(runtime_config["projection_conf_thr"])),
        },

        "detection": {
            "model": detector_key,
            "detections": detections,
            "humerus_found": humerus_detection is not None and not humerus_fallback_used,
            "humerus_fallback_used": bool(humerus_fallback_used),
            "foreign_body_detector_classes": list(FOREIGN_BODY_DETECTOR_CLASS_NAMES),
        },

        "segmentation": {
            "model": seg_key,
            "type": "roi_binary_instance_segmentation",
            "mask_shape": list(seg_mask.shape),
            "num_segmented_instances": len(seg_instances),
            "instances": [
                {k: v for k, v in inst.items() if k != "_mask"}
                for inst in seg_instances
            ],
        },

        "foreign_body": {
            "classifier_model": fb_key,
            "classifier_prob": float(fb_prob),
            "by_classifier": int(fb_check["fb_by_classifier"]),
            "by_detector": int(fb_check["fb_by_detector"]),
            "status": fb_check["status"],
            "confirmed": int(foreign_body_confirmed),
        },

        "detector_fracture_evidence": fracture_evidence,

        "main_bone_fracture": {
            "model": main_key,
            "humerus_detection": humerus_detection,
            "fallback_used": bool(humerus_fallback_used),
            "prob": float(main_bone_fracture_prob),
            "label": int(main_bone_fracture_label),
        },

        "tubercle_fracture": {
            "model": "tubercle",
            "roi_detected": roi_detection is not None,
            "roi_detection": roi_detection,
            "source": tubercle_source,
            "fallback_info": tubercle_fallback_info,
            "prob": None if tubercle_prob is None else float(tubercle_prob),
            "label": tubercle_label,
        },

        "nsa_angle": {
            "model": "nsa_regressor",
            "angle": nsa_angle,
            "acute_angle": nsa_acute_angle,
            "raw_outputs": nsa_raw_outputs,
            "evaluated": nsa_angle is not None,
            "source": angle_source,
            "fallback_info": angle_fallback_info,
        },

        "final": {
            "foreign_body": int(foreign_body_confirmed),
            "main_bone_fracture": int(main_fracture),
            "tubercle_fracture": int(tubercle_fracture),
            "fracture": int(fracture_any),
            "needs_expert_review": int(needs_expert_review),
            "review_reasons": review_reasons,
            "alarm": int(alarm),
        }
    }

    if keep_seg_mask:
        result["_seg_mask"] = seg_mask
        result["_seg_instances"] = seg_instances

    return result


def extract_study_archive(archive_path, extract_root):
    archive_path = Path(archive_path)
    extract_root = Path(extract_root)

    if extract_root.exists():
        shutil.rmtree(extract_root)

    extract_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as zip_file:
        zip_file.extractall(extract_root)

    return extract_root


def find_images(root):
    root = Path(root)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    images = []

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            images.append(path)

    return sorted(images)


def get_study_id_from_archive(archive_path):
    return Path(archive_path).stem


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)

    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    return obj


def strip_runtime_arrays(image_results):
    clean = []

    for result in image_results:
        item = dict(result)
        item.pop("_seg_mask", None)
        item.pop("_seg_instances", None)
        clean.append(item)

    return clean


def aggregate_study_results(image_results, study_id):
    rows = []

    for result in image_results:
        if "error" in result:
            rows.append({
                "study_id": study_id,
                "image": result.get("relative_path"),
                "error": result.get("error"),
                "alarm": 1,
                "needs_expert_review": 1,
            })
            continue

        row = {
            "study_id": study_id,
            "image": result["relative_path"],

            "projection": result["projection"]["label"],
            "projection_confidence": result["projection"]["confidence"],
            "projection_uncertain": result["projection"]["uncertain"],

            "foreign_body_status": result["foreign_body"]["status"],
            "foreign_body_classifier_prob": result["foreign_body"]["classifier_prob"],
            "foreign_body_by_classifier": result["foreign_body"]["by_classifier"],
            "foreign_body_by_detector": result["foreign_body"]["by_detector"],
            "foreign_body_confirmed": result["foreign_body"]["confirmed"],

            "detector_has_fragment_evidence": result["detector_fracture_evidence"]["has_fragment_evidence"],

            "main_bone_fracture_prob": result["main_bone_fracture"]["prob"],
            "main_bone_fracture_label": result["main_bone_fracture"]["label"],
            "main_bone_fallback_used": result["main_bone_fracture"]["fallback_used"],

            "tubercle_fracture_prob": result["tubercle_fracture"]["prob"],
            "tubercle_fracture_label": result["tubercle_fracture"]["label"],
            "tubercle_fracture_source": result["tubercle_fracture"].get("source"),
            "roi_detected": result["tubercle_fracture"]["roi_detected"],

            "nsa_angle": result["nsa_angle"]["angle"],
            "nsa_acute_angle": result["nsa_angle"]["acute_angle"],
            "nsa_angle_evaluated": result["nsa_angle"]["evaluated"],
            "nsa_angle_source": result["nsa_angle"].get("source"),

            "final_fracture": result["final"]["fracture"],
            "final_foreign_body": result["final"]["foreign_body"],
            "needs_expert_review": result["final"]["needs_expert_review"],
            "alarm": result["final"]["alarm"],
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    if len(df) == 0:
        return df, {
            "study_id": study_id,
            "num_images": 0,
            "final": {
                "alarm": 1,
                "needs_expert_review": 1,
            }
        }

    projection_counts = Counter(df.get("projection", pd.Series(dtype=str)).dropna().tolist())

    foreign_body_any = bool((df.get("final_foreign_body", pd.Series(dtype=int)) == 1).any())
    fracture_any = bool((df.get("final_fracture", pd.Series(dtype=int)) == 1).any())
    expert_review_any = bool((df.get("needs_expert_review", pd.Series(dtype=int)) == 1).any())
    alarm_any = bool((df.get("alarm", pd.Series(dtype=int)) == 1).any())

    main_fracture_any = bool((df.get("main_bone_fracture_label", pd.Series(dtype=float)) == 1).any())
    tubercle_fracture_any = bool((df.get("tubercle_fracture_label", pd.Series(dtype=float)) == 1).any())
    fragment_evidence_any = bool((df.get("detector_has_fragment_evidence", pd.Series(dtype=int)) == 1).any())

    angles = df["nsa_angle"].dropna().astype(float).tolist() if "nsa_angle" in df.columns else []
    angle_median = float(np.median(angles)) if len(angles) > 0 else None
    angle_mean = float(np.mean(angles)) if len(angles) > 0 else None

    angle_sources = (
        df["nsa_angle_source"]
        .dropna()
        .astype(str)
        .value_counts()
        .to_dict()
        if "nsa_angle_source" in df.columns
        else {}
    )

    tubercle_sources = (
        df["tubercle_fracture_source"]
        .dropna()
        .astype(str)
        .value_counts()
        .to_dict()
        if "tubercle_fracture_source" in df.columns
        else {}
    )

    review_reasons = []

    for result in image_results:
        if "final" in result and "review_reasons" in result["final"]:
            for reason in result["final"]["review_reasons"]:
                review_reasons.append({
                    "image": result.get("relative_path"),
                    "reason": reason,
                })

    summary = {
        "study_id": study_id,
        "num_images": int(len(image_results)),
        "projection_counts": dict(projection_counts),

        "foreign_body": {
            "confirmed_any": int(foreign_body_any),
        },

        "fracture": {
            "main_bone_any": int(main_fracture_any),
            "tubercle_any": int(tubercle_fracture_any),
            "detector_fragment_evidence_any": int(fragment_evidence_any),
            "any": int(fracture_any),
        },

        "tubercle_fracture": {
            "sources": tubercle_sources,
        },

        "nsa_angle": {
            "values": angles,
            "median": angle_median,
            "mean": angle_mean,
            "sources": angle_sources,
        },

        "final": {
            "alarm": int(alarm_any),
            "needs_expert_review": int(expert_review_any),
            "review_reasons": review_reasons,
        }
    }

    return df, summary
