#!/usr/bin/env python3
"""
Streamlit app for Depth Anything inference on food images

Purpose:
- Let users upload images and visualize estimated depth using Depth Anything.
- Keep all code and optional outputs within the depth model folder.

Usage:
- Run: python -m streamlit run src/models/depth_anything/streamlit_app.py

Notes:
- Downloads pretrained weights on first use (requires network access).
- Works on CPU or CUDA if available.
- Does not write files by default; provides download buttons instead.

Example:
    python -m streamlit run src/models/depth_anything/streamlit_app.py

Resources:
- Model: LiheYoung/depth_anything_{vits|vitb|vitl}14
"""

import io
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

# Local vendor imports
import sys
sys.path.append(str((Path(__file__).resolve().parents[1] / 'src' / 'models' / 'depth_anything')))
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


APP_DIR = Path(__file__).parent


@st.cache_resource(show_spinner=False)
def load_model(encoder: str = "vitb", device_preference: str = "auto"):
    """Load Depth Anything model with caching.

    Args:
        encoder: one of {"vits","vitb","vitl"}
        device_preference: "auto", "cuda", or "cpu"

    Returns:
        model (eval mode), device (str)
    """
    if device_preference == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device_preference == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = f"LiheYoung/depth_anything_{encoder}14"
    model = DepthAnything.from_pretrained(model_id).to(device).eval()
    return model, device


def get_transform() -> Compose:
    return Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])


def infer_depth(model, device: str, image_bgr: np.ndarray) -> np.ndarray:
    """Run depth inference on a single BGR image, return float32 depth map."""
    h, w = image_bgr.shape[:2]
    transform = get_transform()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) / 255.0
    net_input = transform({"image": rgb})["image"]
    net_input = torch.from_numpy(net_input).unsqueeze(0).to(device)

    with torch.no_grad():
        depth = model(net_input)
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = depth.float().cpu().numpy()
    return depth


def colorize_depth(depth: np.ndarray, cmap: str = "INFERNO") -> np.ndarray:
    """Colorize a depth map to a uint8 RGB image using OpenCV colormaps."""
    depth = depth.astype(np.float32)
    dmin, dmax = float(depth.min()), float(depth.max())
    if dmax - dmin < 1e-6:
        norm = np.zeros_like(depth, dtype=np.uint8)
    else:
        norm = ((depth - dmin) / (dmax - dmin) * 255.0).astype(np.uint8)

    cmaps = {
        "INFERNO": cv2.COLORMAP_INFERNO,
        "MAGMA": cv2.COLORMAP_MAGMA,
        "PLASMA": cv2.COLORMAP_PLASMA,
        "VIRIDIS": cv2.COLORMAP_VIRIDIS,
        "TURBO": cv2.COLORMAP_TURBO,
        "JET": cv2.COLORMAP_JET,
    }
    cv2_cmap = cmaps.get(cmap.upper(), cv2.COLORMAP_INFERNO)
    colored = cv2.applyColorMap(norm, cv2_cmap)  # BGR
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return colored


def save_npy_to_buffer(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, array)
    buf.seek(0)
    return buf.read()


def save_png_to_buffer(image_rgb: np.ndarray) -> bytes:
    pil_img = Image.fromarray(image_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def main():
    st.set_page_config(page_title="Depth Anything — Streamlit", layout="wide")
    st.title("Depth Anything — Food Images Depth Explorer")
    st.caption("Upload images to estimate per-pixel depth using a pretrained model.")

    with st.sidebar:
        st.subheader("Settings")
        encoder = st.selectbox("Encoder", options=["vits", "vitb", "vitl"], index=1)
        device_pref = st.selectbox("Device", options=["auto", "cuda", "cpu"], index=0)
        cmap = st.selectbox("Colormap", options=["INFERNO", "MAGMA", "PLASMA", "VIRIDIS", "TURBO", "JET"], index=0)
        st.markdown("---")
        examples_dir = APP_DIR / "assets" / "examples"
        use_example = False
        example_path = None
        if examples_dir.exists():
            use_example = st.checkbox("Use example from assets/examples")
            if use_example:
                example_files = sorted([p for p in examples_dir.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
                if example_files:
                    example_name = st.selectbox("Example image", options=[p.name for p in example_files])
                    example_path = examples_dir / example_name
                else:
                    st.caption("No example images found.")
        st.markdown("---")
        st.caption("Weights are downloaded on first use.")

    with st.spinner("Loading model..."):
        model, device = load_model(encoder, device_pref)
    st.success(f"Model loaded on {device.upper()} with encoder {encoder}")

    uploaded = st.file_uploader(
        "Upload one or more images (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    if use_example and example_path is not None:
        # Process a single example image
        st.markdown(f"### Example: {example_path.name}")
        data = np.fromfile(str(example_path), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            st.warning("Could not read example image.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Original", use_column_width=True)

            with st.spinner("Estimating depth..."):
                depth = infer_depth(model, device, bgr)
            colored = colorize_depth(depth, cmap=cmap)

            with col2:
                st.image(colored, caption=f"Depth ({cmap})", use_column_width=True)

            png_bytes = save_png_to_buffer(colored)
            npy_bytes = save_npy_to_buffer(depth)
            dmin, dmax, dmean = float(depth.min()), float(depth.max()), float(depth.mean())

            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button("Download Depth PNG", data=png_bytes, file_name=f"{example_path.stem}_depth.png", mime="image/png")
            with c2:
                st.download_button("Download Depth NPY", data=npy_bytes, file_name=f"{example_path.stem}_depth.npy", mime="application/octet-stream")
            with c3:
                st.markdown(f"Min: {dmin:.4f} • Max: {dmax:.4f} • Mean: {dmean:.4f}")

    elif uploaded:
        for i, file in enumerate(uploaded, start=1):
            st.markdown(f"### Image {i}: {file.name}")
            data = np.frombuffer(file.read(), dtype=np.uint8)
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if bgr is None:
                st.warning("Could not read image.")
                continue

            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), caption="Original", use_column_width=True)

            with st.spinner("Estimating depth..."):
                depth = infer_depth(model, device, bgr)
            colored = colorize_depth(depth, cmap=cmap)

            with col2:
                st.image(colored, caption=f"Depth ({cmap})", use_column_width=True)

            # Downloads
            png_bytes = save_png_to_buffer(colored)
            npy_bytes = save_npy_to_buffer(depth)
            dmin, dmax, dmean = float(depth.min()), float(depth.max()), float(depth.mean())

            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button("Download Depth PNG", data=png_bytes, file_name=f"{Path(file.name).stem}_depth.png", mime="image/png")
            with c2:
                st.download_button("Download Depth NPY", data=npy_bytes, file_name=f"{Path(file.name).stem}_depth.npy", mime="application/octet-stream")
            with c3:
                st.markdown(f"Min: {dmin:.4f} • Max: {dmax:.4f} • Mean: {dmean:.4f}")

    else:
        st.info("Upload image files to begin.")


if __name__ == "__main__":
    main()
