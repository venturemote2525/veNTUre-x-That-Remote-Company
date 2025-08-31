#!/usr/bin/env python3
"""
Streamlit app for Food Mask R-CNN segmentation inference using latest trained checkpoint.

Launch:
  python -m streamlit run ui/segmentation_app.py
"""

import io
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st
import torch
import torchvision
import cv2
from PIL import Image

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PMR_DIR = REPO_ROOT / "src" / "models" / "pytorch_mask_rcnn"
sys.path.append(str(PMR_DIR))
sys.path.append(str(PMR_DIR / "utils"))

from train_maskrcnn_food import FoodMaskRCNNTrainer  # type: ignore
from utils.config import FoodMaskRCNNConfig  # type: ignore


def list_checkpoint_dirs() -> List[Path]:
    candidates = [
        REPO_ROOT / "src" / "training" / "checkpoints",
        REPO_ROOT / "checkpoints",
    ]
    found = []
    for base in candidates:
        if base.exists():
            for sub in base.iterdir():
                if sub.is_dir() and sub.name.startswith("swiss_7class_"):
                    found.append(sub)
    return sorted(found)


def pick_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    if (ckpt_dir / "best_checkpoint.pth").exists():
        return ckpt_dir / "best_checkpoint.pth"
    if (ckpt_dir / "latest_checkpoint.pth").exists():
        return ckpt_dir / "latest_checkpoint.pth"
    epoch_ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pth"))
    return epoch_ckpts[-1] if epoch_ckpts else None


@st.cache_resource(show_spinner=False)
def load_model_from_checkpoint(ckpt_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(ckpt_path, map_location=device)

    cfg_dict = checkpoint.get("config", {})
    name = cfg_dict.get("name", "inference_swiss_7class")
    backbone = cfg_dict.get("backbone", "resnet50")
    class_names = cfg_dict.get("class_names", ["carb", "meat", "vegetable", "others"])

    cfg = FoodMaskRCNNConfig(
        name=name,
        class_names=class_names,
        backbone=backbone,
        epochs=1,
        batch_size=1,
        image_min_size=cfg_dict.get("image_min_size", 640),
        image_max_size=cfg_dict.get("image_max_size", 800),
        checkpoint_dir=str(Path(ckpt_path).parent),
        log_dir=str(Path(ckpt_path).parent.parent / "logs" / name),
    )

    trainer = FoodMaskRCNNTrainer(cfg)
    trainer.build_model()
    state = checkpoint.get("model_state_dict") or checkpoint
    trainer.model.load_state_dict(state)
    trainer.model.eval().to(device)
    return trainer.model, device, class_names


def tensor_from_image(image: np.ndarray) -> torch.Tensor:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image_rgb).float() / 255.0
    tensor = tensor.permute(2, 0, 1).contiguous()
    return tensor


def draw_predictions(image_bgr: np.ndarray, pred: dict, class_names: List[str], conf_thr: float = 0.5) -> np.ndarray:
    out = image_bgr.copy()
    boxes = pred.get("boxes")
    scores = pred.get("scores")
    labels = pred.get("labels")
    masks = pred.get("masks")
    if boxes is None or boxes.numel() == 0:
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    boxes = boxes.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    keep = scores >= conf_thr
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    overlay = out.copy()
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = box.astype(int)
        color = (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        # Labels from Mask R-CNN are 1..N (0 = background). Map to class_names (0..N-1)
        if label <= 0:
            name = "background"
        else:
            idx = int(label) - 1
            name = class_names[idx] if 0 <= idx < len(class_names) else f"class_{label}"
        cv2.putText(out, f"{name}:{score:.2f}", (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        if masks is not None and len(masks) > i:
            m = masks[i].detach().cpu().numpy()
            if m.ndim == 3:
                m = m[0]
            m = (m > 0.5).astype(np.uint8)
            if m.sum() > 0:
                colored = np.zeros_like(out)
                colored[:, :, 2] = 255  # red
                mask3 = np.dstack([m]*3)
                overlay = np.where(mask3, cv2.addWeighted(colored, 0.4, overlay, 0.6, 0), overlay)

    out = np.where(overlay != out, overlay, out)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def resize_for_inference(bgr: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    scale = min(1.0, float(max_dim) / max(h, w))
    if scale < 1.0:
        nh, nw = int(round(h * scale)), int(round(w * scale))
        bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return bgr


def main():
    st.set_page_config(page_title="Food Segmentation — Streamlit", layout="wide")
    st.title("Food Segmentation — Mask R-CNN Inference")
    st.caption("Uploads → predictions using the latest trained checkpoint")

    ckpt_dirs = list_checkpoint_dirs()
    if not ckpt_dirs:
        st.error("No checkpoint directories found. Train a model first.")
        return

    default_dir_idx = 0
    for i, d in enumerate(ckpt_dirs):
        if d.name.endswith("resnet50"):
            default_dir_idx = i
            break

    choice = st.selectbox("Checkpoint directory", options=[str(d) for d in ckpt_dirs], index=default_dir_idx)
    ckpt_dir = Path(choice)
    ckpt_path = pick_checkpoint(ckpt_dir)
    if not ckpt_path:
        st.error("No checkpoint files found in the selected directory.")
        return

    conf_thr = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
    max_dim = st.slider("Max image dimension", 256, 2048, 1024, 64)

    with st.spinner(f"Loading model from {ckpt_path.name} ..."):
        model, device, class_names = load_model_from_checkpoint(str(ckpt_path))
    st.success(f"Model loaded on {device.upper()} with classes: {class_names}")

    files = st.file_uploader("Upload images (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if not files:
        st.info("Upload one or more images to run inference.")
        return

    for i, file in enumerate(files, start=1):
        st.markdown(f"### Image {i}: {file.name}")
        data = np.frombuffer(file.read(), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            st.warning("Could not read image.")
            continue

        bgr = resize_for_inference(bgr, max_dim)
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Original", use_column_width=True)

        with st.spinner("Running inference..."):
            inp = tensor_from_image(bgr).unsqueeze(0).to(device)
            with torch.no_grad():
                preds = model(inp)
            pred = preds[0]
            vis = draw_predictions(bgr, pred, class_names, conf_thr=conf_thr)

        with col2:
            st.image(vis, caption="Predictions", use_column_width=True)


if __name__ == "__main__":
    main()
