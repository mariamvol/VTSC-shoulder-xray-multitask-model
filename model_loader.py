from pathlib import Path

import torch
import torch.nn as nn

from torchvision.models import resnet18, resnet34, densenet121
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import segmentation_models_pytorch as smp


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MAIN_BONE_IMG_SIZE = 224
MAIN_BONE_THRESHOLD = 0.50

TUBERCLE_IMG_SIZE = 384
TUBERCLE_THRESHOLD = 0.50

SEG_IMG_SIZE = 256
SEG_THRESHOLD = 0.50
SEG_ROI_MARGIN = 0.12

ROI_DETECTOR_IMG_MAX = 1024
ROI_DETECTOR_INTERNAL_SCORE_THR = 0.001
ROI_DETECTIONS_PER_IMG = 10

NSA_IMG_SIZE = 384


def safe_torch_load(source, map_location="cpu"):

    if isinstance(source, (str, Path)):
        source = Path(source)

        if not source.exists():
            raise FileNotFoundError(f"Файл не найден: {source}")

        try:
            return torch.load(source, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(source, map_location=map_location)

    return source


def get_source_name(source):
    if isinstance(source, (str, Path)):
        return Path(source).name
    return "<from_bundle>"


def clean_state_dict_keys(state_dict):
    cleaned = {}

    for key, value in state_dict.items():
        new_key = key

        for prefix in ["module.", "model.", "net."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]

        cleaned[new_key] = value

    return cleaned


def get_state_dict_from_ckpt(ckpt):
    if isinstance(ckpt, nn.Module):
        return ckpt

    if not isinstance(ckpt, dict):
        raise ValueError(f"Неизвестный формат checkpoint: {type(ckpt)}")

    for key in ["model_state", "model_state_dict", "state_dict", "model", "net"]:
        if key in ckpt:
            return ckpt[key]

    if all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt

    raise ValueError(f"Не удалось найти state_dict. Ключи: {list(ckpt.keys())}")


def build_resnet_classifier(arch, num_outputs):
    if arch == "resnet18":
        model = resnet18(weights=None)
    elif arch == "resnet34":
        model = resnet34(weights=None)
    else:
        raise ValueError(f"Неизвестная ResNet-архитектура: {arch}")

    model.fc = nn.Linear(model.fc.in_features, num_outputs)
    return model


def load_resnet_classifier_from_ckpt(source, device):

    ckpt = safe_torch_load(source, map_location="cpu")

    meta = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}

    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        state_dict = get_state_dict_from_ckpt(ckpt)

    state_dict = clean_state_dict_keys(state_dict)

    arch = meta.get("arch", "resnet18")
    num_outputs = int(state_dict["fc.weight"].shape[0])

    model = build_resnet_classifier(arch, num_outputs)
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    meta_out = {
        "source": get_source_name(source),
        "arch": arch,
        "img_size": int(meta.get("img_size", 512)),
        "threshold": float(meta.get("threshold", 0.5)),
        "mean": meta.get("mean", IMAGENET_MEAN),
        "std": meta.get("std", IMAGENET_STD),
        "classes": meta.get("classes"),
        "task": meta.get("task"),
        "projection": meta.get("projection"),
        "best_val_auc": meta.get("best_val_auc"),
        "num_outputs": num_outputs,
        "square_pad": False,
        "raw_meta": meta,
    }

    return model, meta_out


def load_main_bone_densenet121(source, device):

    ckpt = safe_torch_load(source, map_location="cpu")

    state_dict = ckpt["model_state"]

    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 1)

    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    meta = {
        "source": get_source_name(source),
        "arch": "densenet121",
        "img_size": MAIN_BONE_IMG_SIZE,
        "threshold": MAIN_BONE_THRESHOLD,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "num_outputs": 1,
        "square_pad": False,
        "input_source": "crop_from_humerus_box_detector",
        "best_epoch": ckpt.get("best_epoch"),
        "best_auc_study": ckpt.get("best_auc_study"),
        "metrics": ckpt.get("metrics"),
        "hpo_params": ckpt.get("hpo_params"),
    }

    return model, meta


def load_tubercle_classifier(source, device):

    ckpt = safe_torch_load(source, map_location="cpu")

    model = resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)

    model.load_state_dict(ckpt["model_state"], strict=True)

    model.to(device)
    model.eval()

    meta = {
        "source": get_source_name(source),
        "arch": "resnet34",
        "img_size": TUBERCLE_IMG_SIZE,
        "threshold": TUBERCLE_THRESHOLD,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "num_outputs": 1,
        "square_pad": True,
        "input_source": "crop_from_roi_detector_or_tubercle_detector_fallback",
        "best_epoch": ckpt.get("best_epoch"),
        "best_auc_study": ckpt.get("best_auc_study"),
        "metrics": ckpt.get("metrics"),
        "hpo_params": ckpt.get("hpo_params"),
    }

    return model, meta


class NSARegressor(nn.Module):

    def __init__(self):
        super().__init__()

        backbone = resnet18(weights=None)
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def load_nsa_regressor(source, device):
    ckpt = safe_torch_load(source, map_location="cpu")

    model = NSARegressor()
    model.load_state_dict(ckpt["model"], strict=True)

    model.to(device)
    model.eval()

    meta = {
        "source": get_source_name(source),
        "arch": "resnet18_backbone_custom_head",
        "img_size": NSA_IMG_SIZE,
        "num_outputs": 2,
        "input_source": "crop_from_roi_detector_or_bone_detector_fallback",
        "normalization": "(x - 0.5) / 0.5",
        "output": "cos_2theta_sin_2theta",
    }

    return model, meta


def build_fasterrcnn_resnet50_fpn_v2_detector(num_classes, img_size):
    model = fasterrcnn_resnet50_fpn_v2(
        weights=None,
        weights_backbone=None,
        min_size=img_size,
        max_size=img_size,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def load_bone_detector(source, device):

    ckpt = safe_torch_load(source, map_location="cpu")

    meta = ckpt["meta"]
    state_dict = ckpt["model"]

    num_classes = len(meta["classes"]) + 1
    img_size = int(meta["img_size"])

    model = build_fasterrcnn_resnet50_fpn_v2_detector(
        num_classes=num_classes,
        img_size=img_size,
    )

    model.load_state_dict(state_dict, strict=True)

    model.roi_heads.score_thresh = float(meta.get("score_thr", 0.3))
    model.roi_heads.nms_thresh = float(meta.get("nms_thr", 0.2))
    model.roi_heads.detections_per_img = int(meta.get("detections_per_img", 50))

    model.to(device)
    model.eval()

    label_map = {int(k): v for k, v in meta["classes"].items()}

    meta_out = {
        "source": get_source_name(source),
        "arch": "fasterrcnn_resnet50_fpn_v2",
        "img_size": img_size,
        "label_map": label_map,
        "score_thr": model.roi_heads.score_thresh,
        "nms_thr": model.roi_heads.nms_thresh,
        "detections_per_img": model.roi_heads.detections_per_img,
        "raw_meta": meta,
    }

    return model, meta_out


def build_roi_fasterrcnn_detector(num_classes=2):

    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def load_roi_detector(source, device):
    ckpt = safe_torch_load(source, map_location="cpu")

    model = build_roi_fasterrcnn_detector(num_classes=2)
    model.load_state_dict(ckpt["model"], strict=True)

    model.roi_heads.score_thresh = ROI_DETECTOR_INTERNAL_SCORE_THR
    model.roi_heads.detections_per_img = ROI_DETECTIONS_PER_IMG

    model.to(device)
    model.eval()

    meta = {
        "source": get_source_name(source),
        "arch": "fasterrcnn_resnet50_fpn",
        "num_classes": 2,
        "label_map": {1: "roi"},
        "det_img_max": ROI_DETECTOR_IMG_MAX,
        "internal_score_thr": ROI_DETECTOR_INTERNAL_SCORE_THR,
        "detections_per_img": ROI_DETECTIONS_PER_IMG,
        "input_preprocess": "grayscale_to_rgb_resize_max_side_1024_to_tensor",
    }

    return model, meta


def build_roi_unet_segmentation():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    return model


def load_roi_segmentation_model(source, device):
    """
    ROI-сегментация:
    SMP Unet, encoder=resnet34, classes=1, IMG_SIZE=256.
    """
    ckpt = safe_torch_load(source, map_location="cpu")

    model = build_roi_unet_segmentation()
    model.load_state_dict(ckpt["model_state"], strict=True)

    model.to(device)
    model.eval()

    meta = {
        "source": get_source_name(source),
        "arch": "smp.Unet",
        "encoder": "resnet34",
        "classes": 1,
        "img_size": SEG_IMG_SIZE,
        "threshold": SEG_THRESHOLD,
        "roi_margin": SEG_ROI_MARGIN,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "segmentation_type": "roi_binary_instance_segmentation",
        "projection": ckpt.get("projection"),
        "epoch": ckpt.get("epoch"),
        "best_dice": ckpt.get("best_dice"),
    }

    return model, meta


def load_all_models_from_sources(sources, device):
    models = {}
    metas = {}

    models["projection"], metas["projection"] = load_resnet_classifier_from_ckpt(
        sources["projection"], device
    )

    models["foreign_body_D"], metas["foreign_body_D"] = load_resnet_classifier_from_ckpt(
        sources["foreign_body_D"], device
    )

    models["foreign_body_S"], metas["foreign_body_S"] = load_resnet_classifier_from_ckpt(
        sources["foreign_body_S"], device
    )

    models["detector_D"], metas["detector_D"] = load_bone_detector(
        sources["detector_D"], device
    )

    models["detector_S"], metas["detector_S"] = load_bone_detector(
        sources["detector_S"], device
    )

    models["seg_D"], metas["seg_D"] = load_roi_segmentation_model(
        sources["seg_D"], device
    )

    models["seg_S"], metas["seg_S"] = load_roi_segmentation_model(
        sources["seg_S"], device
    )

    models["main_bone_D"], metas["main_bone_D"] = load_main_bone_densenet121(
        sources["main_bone_D"], device
    )

    models["main_bone_S"], metas["main_bone_S"] = load_main_bone_densenet121(
        sources["main_bone_S"], device
    )

    models["roi_detector"], metas["roi_detector"] = load_roi_detector(
        sources["roi_detector"], device
    )

    models["tubercle"], metas["tubercle"] = load_tubercle_classifier(
        sources["tubercle"], device
    )

    models["nsa_regressor"], metas["nsa_regressor"] = load_nsa_regressor(
        sources["nsa_regressor"], device
    )

    return models, metas


def load_all_models_from_bundle(bundle_path, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    bundle = safe_torch_load(bundle_path, map_location="cpu")

    if "weights" not in bundle:
        raise KeyError("В bundle нет ключа 'weights'.")

    models, metas = load_all_models_from_sources(bundle["weights"], device)

    return models, metas, bundle


def load_pipeline(bundle_path, device=None):
    models, metas, bundle = load_all_models_from_bundle(bundle_path, device=device)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ctx = {
        "models": models,
        "metas": metas,
        "bundle": bundle,
        "device": device,
    }

    return ctx
