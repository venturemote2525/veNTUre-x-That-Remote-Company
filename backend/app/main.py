#!/usr/bin/env python3
"""
Complete Food Analysis API - Working implementation with all features
"""
import sys, os, io
from pathlib import Path
import base64
from io import BytesIO
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
# Removed torch dependency to simplify runtime
import cv2
import onnxruntime as ort
# from torchvision import transforms  # Replaced with lightweight preprocessing
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
# albumentations not used in this module
try:
    import albumentations as A  # noqa: F401
except Exception:
    A = None
import math
from skimage.measure import regionprops
import base64
from io import BytesIO

# Add paths
API_ROOT = Path(__file__).resolve().parent
sys.path.append(str(API_ROOT))

from food_ai.storage.food_class_names import FOOD_CLASS_NAMES
from food_ai.storage.nutrition_database import NutritionDatabase
# try:
#     from reference.core.reference_scale import HeuristicReferenceObjectDetector  # optional hybrid heuristic
# except Exception:
#     HeuristicReferenceObjectDetector = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI
app = FastAPI(title="Food Analysis API", version="2.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
MODELS = {}

# Utensil reference sizes (in mm)
UTENSIL_SIZES = {
    0: {"name": "chopsticks", "length_mm": 230.0, "width_mm": 8.0},  # Standard Chinese chopsticks
    1: {"name": "spoon", "length_mm": 160.0, "width_mm": 35.0},     # Dinner/dessert spoon average (aligned with non-API)
    2: {"name": "fork", "length_mm": 180.0, "width_mm": 25.0},      # Dinner fork (aligned with non-API)
}

# YOLO class ordering for ONNX utensil detector (3-class model)
YOLO_UTENSIL_LABELS = ["spoon", "fork"]

class AnalyzeRequest(BaseModel):
    bucket: str  # Supabase storage bucket name
    path: str    # Path to image in bucket
    image_base64: Optional[str] = None
    confidence_threshold: float = 0.2
    enable_volume: bool = True
    enable_nutrition: bool = True
    dedup_overlap_threshold: float = 0.3  # Enable deduplication by default to prevent over-counting
    # Depth/volume options
    baseline_method: str = "food_95th"
    # Utensil tuning (aligned with non-API which uses conf=0.5)
    utensil_conf: float = 0.6  # Raised from 0.01 to match non-API and reduce false positives
    utensil_nms_iou: float = 0.3
    utensil_min_box_px: int = 5
    utensil_promote_top_n: int = 2
    spoon_length_mm: float = 160.0  # Dinner/dessert spoon average (aligned with non-API)
    fork_length_mm: float = 180.0  # Dinner fork (aligned with non-API)
    # Synthetic fallback scale
    synthetic_px_per_mm_1024: float = 3.5
    prefer_synthetic_first: bool = False  # Use real utensil scale when available!
    # Classification threshold for per-detection specific class
    classification_conf_threshold: float = 0.40
    # Minimum weighted score for segment inclusion in weighted classification
    min_weighted_score_pct: float = 2.0

class AnalyzeResponse(BaseModel):
    status: str
    classification: dict  # Kept for backwards compatibility
    segmentation: dict
    utensils: list = []
    volume_estimates: dict = {}
    nutrition: dict = {}
    summary: dict = {}
    visuals: dict = {}
    classification_methods: dict = {}  # New: two-method comparison results

@app.get("/")
def root():
    return {"message": "Hello from FastAPI + Supabase!"}

@app.on_event("startup")
async def load_models():
    """Load all models on startup"""
    logger.info("Loading models...")

    # Load ViT classification model (ONNX) - optional for per-detection classification
    vit_path = API_ROOT / "food_ai" / "models" / "classification" / "vit_food_classifier.onnx"
    try:
        MODELS['vit'] = ort.InferenceSession(str(vit_path))
        logger.info(f"[OK] ViT model loaded: {vit_path}")
    except Exception as e:
        logger.warning(f"[WARN] VIT model failed to load (per-detection classification disabled): {e}")
        logger.info("[INFO] API will use segmentation broad classes only for nutrition")

    # Load SegFormer segmentation model using wrapper (working Streamlit code)
    from food_ai.segmentation_wrapper import get_segmentation_model
    MODELS['segformer'] = get_segmentation_model()
    logger.info("[OK] SegFormer model loaded via wrapper")

    # Load utensil detection model (YOLO ONNX)
    utensil_path = API_ROOT / "food_ai" / "models" / "utensil_detection" / "utensil_detector_converted.onnx"
    if utensil_path.exists():
        MODELS['utensil'] = ort.InferenceSession(str(utensil_path))
        logger.info(f"[OK] Utensil detector loaded: {utensil_path}")
    # Initialize optional heuristic detector
    # if HeuristicReferenceObjectDetector is not None:
    #     try:
    #         MODELS['utensil_hybrid'] = HeuristicReferenceObjectDetector(['spoon', 'fork'])
    #         logger.info("[OK] Heuristic utensil detector initialized")
    #     except Exception as e:
    #         logger.warning(f"[UTENSIL] Heuristic init failed: {e}")

    # Load depth model
    depth_path = API_ROOT / "food_ai" / "models" / "volume" / "depth_anything_vitb.onnx"
    if depth_path.exists():
        MODELS['depth'] = ort.InferenceSession(str(depth_path))
        logger.info(f"[OK] Depth model loaded: {depth_path}")

    # Initialize nutrition database
    MODELS['nutrition_db'] = NutritionDatabase()
    logger.info("[OK] Nutrition database initialized")

service_supabase: Client = create_client(
    os.getenv("SUPABASE_URL"), 
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # This bypasses RLS
)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": list(MODELS.keys()),
        "model_count": len(MODELS)
    }

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(request: AnalyzeRequest):
    """
    Complete food image analysis with nutrition calculation
    """
    try:
        res = service_supabase.storage.from_(request.bucket).download(request.path)
        if res is None and request.image_base64:
            # fallback to decode base64
            img_bytes = base64.b64decode(request.image_base64.split(",")[-1])
            image = Image.open(BytesIO(img_bytes)).convert("RGB")
        elif res is None:
            raise HTTPException(status_code=404, detail="Image not found")
        else:
            image = Image.open(io.BytesIO(res)).convert("RGB")

        # # Decode image
        # if request.image_base64.startswith('data:image'):
        #     img_data = request.image_base64.split(',')[1]
        # else:
        #     img_data = request.image_base64

        # img_bytes = base64.b64decode(img_data)
        # image = Image.open(BytesIO(img_bytes)).convert('RGB')
        image_array = np.array(image)

        logger.info(f"Processing image: {image_array.shape}")

        # 1. Run classification
        classification_result = classify_food(MODELS['vit'], image_array)
        logger.info(f"Classification: {classification_result['top_prediction']['class_name']} ({classification_result['top_prediction']['confidence']:.3f})")
        print(f"Classification: {classification_result['top_prediction']['class_name']} ({classification_result['top_prediction']['confidence']:.3f})") # e.g. chicken rice (0.715)

        # 2. Run segmentation (per-instance, probability-thresholded)
        seg_threshold = request.confidence_threshold
        segmentation_result = segment_food(MODELS['segformer'], image_array, seg_threshold)

        # Duplicate segment filtering (enabled by default to prevent over-counting)
        if request.dedup_overlap_threshold > 0.0:
            before = len(segmentation_result.get('detections', []))
            segmentation_result['detections'] = deduplicate_overlapping_segments(
                segmentation_result.get('detections', []),
                overlap_threshold=request.dedup_overlap_threshold
            )
            after = len(segmentation_result.get('detections', []))
            if after != before:
                logger.info(f"Segmentation: deduplicated {before} -> {after} (removed {before-after} overlapping regions)")

        logger.info(f"Segmentation: {len(segmentation_result['detections'])} detections")

        # 3. Run per-detection VIT classification (before volume calculation) - if VIT model loaded
        if segmentation_result['detections'] and 'vit' in MODELS:
            logger.info(f"[VIT] Running per-detection classification on {len(segmentation_result['detections'])} segments")
            for idx, det in enumerate(segmentation_result['detections']):
                seg_cls = classify_segment(MODELS['vit'], image_array, det)
                det['specific_food'] = seg_cls.get('class_name')
                det['specific_confidence'] = seg_cls.get('confidence', 0.0)
        elif segmentation_result['detections'] and 'vit' not in MODELS:
            logger.info("[VIT] Skipping per-detection classification (VIT model not loaded)")

        # 4. Detect utensils for scaling
        utensil_detections = []
        pixel_to_mm_ratio = None
        if request.enable_volume:
            utensil_detections = []
            # Try YOLO ONNX detector first
            if 'utensil' in MODELS:
                utensil_detections = detect_utensils(
                    MODELS['utensil'],
                    image_array,
                    conf_thres=float(request.utensil_conf),
                    nms_iou=float(request.utensil_nms_iou),
                    min_box_px=int(request.utensil_min_box_px),
                    promote_top_n=int(request.utensil_promote_top_n),
                )
            # Fallback: heuristic/hybrid if no YOLO detection
            if (not utensil_detections) and ('utensil_hybrid' in MODELS):
                try:
                    bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                    hy = MODELS['utensil_hybrid'].detect(bgr)
                    for d in hy:
                        bbox = getattr(d, 'bbox', None)
                        if not bbox:
                            continue
                        x1,y1,x2,y2 = map(int, bbox)
                        utensil_detections.append({
                            'class_id': None,
                            'class_name': getattr(d, 'object_type', 'utensil'),
                            'confidence': float(getattr(d, 'confidence', 0.5)),
                            'bbox': [x1,y1,x2,y2],
                            'width_px': int(max(0, x2-x1)),
                            'height_px': int(max(0, y2-y1)),
                        })
                except Exception as e:
                    logger.warning(f"[UTENSIL] Heuristic detection failed: {e}")
            if utensil_detections:
                pixel_to_mm_ratio = calculate_pixel_to_mm_ratio(
                    utensil_detections,
                    spoon_len_mm=float(request.spoon_length_mm),
                    fork_len_mm=float(request.fork_length_mm),
                    utensil_conf_threshold=float(request.utensil_conf)
                )
                if pixel_to_mm_ratio:
                    logger.info(f"Utensils detected: {len(utensil_detections)}, scale: {pixel_to_mm_ratio:.4f} mm/px")
                else:
                    logger.info(f"Utensils detected: {len(utensil_detections)}, but no valid scale calculated")
            # Synthetic fallback for scale (parity with non-API)
            synth_ratio = synthetic_pixel_to_mm_ratio(image_array, px_per_mm_1024=float(request.synthetic_px_per_mm_1024))
            if request.prefer_synthetic_first and synth_ratio:
                pixel_to_mm_ratio = synth_ratio
                logger.info(f"[UTENSIL] Prefer synthetic: {pixel_to_mm_ratio:.4f} mm/px")
            elif pixel_to_mm_ratio is None and synth_ratio:
                pixel_to_mm_ratio = synth_ratio
                logger.info(f"[UTENSIL] Using synthetic fallback: {pixel_to_mm_ratio:.4f} mm/px")

        # Pre-compute depth map (used for volume and visuals)
        depth_map = None
        if MODELS.get('depth') is not None:
            depth_map = compute_depth_map(MODELS['depth'], image_array)

        # 4. Calculate volumes (depth_map with food_95th baseline only)
        volume_estimates = {}
        if request.enable_volume and segmentation_result['detections']:
            logger.info(f"[DEBUG] About to calculate volumes for {len(segmentation_result['detections'])} detections")
            # Check if detections have masks
            masks_present = sum(1 for d in segmentation_result['detections'] if 'mask' in d and d['mask'] is not None)
            logger.info(f"[DEBUG] Detections with masks: {masks_present}/{len(segmentation_result['detections'])}")

            if depth_map is not None:
                volume_estimates = calculate_volumes_depth_map(
                    image_array,
                    segmentation_result['detections'],
                    pixel_to_mm_ratio,
                    depth_map,
                    baseline_method=request.baseline_method,
                    depth_upscale_factor=1.0,
                    vit_classifications=None,  # per-det VIT is stored on detections directly
                    classification_confidence_threshold=float(request.classification_conf_threshold)
                )
                logger.info(f"Volume calculated for {len(volume_estimates)} items")
            else:
                logger.error("[VOL] Depth model not available, volume estimation disabled")
                volume_estimates = {}

        # 5. Get nutrition information
        nutrition_result = {}
        summary = {}
        if request.enable_nutrition:
            nutrition_result = calculate_nutrition(
                segmentation_result['detections'],
                classification_result,
                volume_estimates,
                MODELS['nutrition_db'],
                classification_confidence_threshold=float(request.classification_conf_threshold)
            )
            summary = nutrition_result.get('summary', {})
            logger.info(f"Nutrition: {summary.get('total_calories', 0):.1f} kcal")

        # 6. Calculate weighted classification (two-method comparison)
        classification_methods = {}
        if segmentation_result['detections'] and 'vit' in MODELS:
            try:
                weighted_result = calculate_weighted_classification(
                    image_array,
                    segmentation_result['detections'],
                    classification_result['top_prediction'],
                    min_weighted_score_pct=float(request.min_weighted_score_pct)
                )

                classification_methods = {
                    "full_image": {
                        "class": classification_result['top_prediction']['class_name'],
                        "confidence": float(classification_result['top_prediction']['confidence'])
                    },
                    "weighted_segments": {
                        "class": weighted_result['best_segment']['class'] if weighted_result['best_segment'] else None,
                        "confidence": float(weighted_result['weighted_confidence']),
                        "breakdown": [
                            f"{seg['class'].replace('_', ' ').title()} ({seg['weighted_score_pct']:.1f}%)"
                            for seg in weighted_result['all_segments']
                        ]
                    },
                    "recommendation": weighted_result['recommendation'],
                    "multi_segment_bonus": weighted_result['multi_segment_bonus'],
                    "num_big_segments": weighted_result['num_big_segments']
                }

                logger.info(f"[CLASSIFICATION] Full image: {classification_result['top_prediction']['class_name']} ({classification_result['top_prediction']['confidence']:.3f})")
                logger.info(f"[CLASSIFICATION] Weighted segments: {weighted_result['weighted_confidence']:.3f}, Recommendation: {weighted_result['recommendation']}")
            except Exception as e:
                logger.warning(f"[CLASSIFICATION] Weighted classification failed: {e}")
                classification_methods = {}

        # Visuals (segmentation overlay + depth map preview)
        visuals = {}
        try:
            overlay_img = create_segmentation_overlay(image_array, segmentation_result.get('detections', []), utensil_detections)
            visuals['segmentation_overlay_png'] = encode_png_base64(overlay_img)
            if depth_map is not None:
                depth_vis = visualize_depth_map(depth_map)
                visuals['depth_map_png'] = encode_png_base64(depth_vis)
            # Utensil debug overlay
            if 'utensil' in MODELS:
                debug_cands = detect_utensil_candidates_debug(MODELS['utensil'], image_array, top_k=20)
                if debug_cands:
                    utensil_overlay = image_array.copy()
                    # draw debug candidates in cyan, final detections in yellow
                    for c in debug_cands:
                        x1,y1,x2,y2 = c['bbox']
                        cv2.rectangle(utensil_overlay, (x1,y1), (x2,y2), (0,255,255), 1)
                    for u in utensil_detections or []:
                        if u.get('bbox') and len(u['bbox'])==4:
                            x1,y1,x2,y2 = map(int, u['bbox'])
                            cv2.rectangle(utensil_overlay, (x1,y1), (x2,y2), (255,215,0), 2)
                    visuals['utensil_overlay_png'] = encode_png_base64(utensil_overlay)
        except Exception as e:
            logger.warning(f"[VISUALS] Failed to create visuals: {e}")

        # Clean masks for JSON serialization (after volume calculation)
        for detection in segmentation_result.get('detections', []):
            detection.pop('mask', None)

        return {
            "status": "success",
            "classification": classification_result,  # Kept for backwards compatibility
            "segmentation": segmentation_result,
            "utensils": utensil_detections,
            "volume_estimates": volume_estimates,
            "nutrition": nutrition_result,
            "summary": summary,
            "visuals": visuals,
            "classification_methods": classification_methods  # New: two-method comparison
        }

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def classify_food(session, image_array):
    """Classification using ViT ONNX model without torch/torchvision.

    - Resize to 224x224
    - Convert to NCHW float32
    - Normalize with ImageNet mean/std
    - ONNX inference and numpy softmax
    """
    # Ensure PIL Image
    if isinstance(image_array, np.ndarray):
        image = Image.fromarray(image_array)
    else:
        image = image_array

    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize
    image = image.resize((224, 224), Image.Resampling.BILINEAR)

    # To numpy CHW float32 normalized
    arr = np.asarray(image).astype(np.float32) / 255.0  # HWC, [0,1]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    chw = np.transpose(arr, (2, 0, 1))  # CHW
    input_tensor = np.expand_dims(chw, axis=0).astype(np.float32)  # NCHW

    # Run ONNX inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    logits = outputs[0][0]  # 1D array

    # Numpy softmax
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    probs = exp / np.sum(exp)

    # Top-k
    top_k = min(10, len(FOOD_CLASS_NAMES))
    top_indices = np.argsort(-probs)[:top_k]
    top_probs = probs[top_indices]

    results = []
    for i in range(top_k):
        idx = int(top_indices[i])
        prob = float(top_probs[i])
        class_name = FOOD_CLASS_NAMES[idx] if idx < len(FOOD_CLASS_NAMES) else f"class_{idx}"
        results.append({
            "class_name": class_name,
            "confidence": prob,
            "index": idx
        })

    return {
        "top_prediction": results[0],
        "top_5": results[:5],
        "all_predictions": results
    }

def classify_segment(session, image_array, detection):
    """
    Compatibility wrapper returning keys expected by analyze_image.
    """
    try:
        result = classify_detection(session, image_array, detection)
        if not result:
            return {"class_name": None, "confidence": 0.0}
        return {
            "class_name": result.get("class") or result.get("class_name"),
            "confidence": float(result.get("confidence", 0.0)),
            "index": result.get("index")
        }
    except Exception:
        return {"class_name": None, "confidence": 0.0}

def classify_detection(session, image_array, detection):
    """Run VIT classification on a single detection segment.

    Args:
        session: ONNX VIT model session
        image_array: Full RGB image
        detection: Detection dict with bbox and mask

    Returns:
        dict with 'class', 'confidence', or None if classification fails
    """
    try:
        # Get detection bounds
        x1, y1, x2, y2 = detection['bbox']
        mask = detection.get('mask')
        h, w = image_array.shape[:2]

        # Ensure bounds are valid
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))

        # Create masked region (crop + apply mask if available)
        if mask is not None:
            # Resize mask to image dimensions if needed
            if mask.shape[:2] != (h, w):
                mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                mask_resized = mask.astype(np.uint8)

            # Create RGBA masked image
            masked_image = np.zeros((h, w, 4), dtype=np.uint8)
            masked_image[:, :, :3] = image_array
            masked_image[:, :, 3] = mask_resized * 255

            # Crop to bbox
            cropped_masked = masked_image[y1:y2, x1:x2]

            # Convert to RGB with white background
            crop_h, crop_w = cropped_masked.shape[:2]
            white_bg = np.ones((crop_h, crop_w, 3), dtype=np.uint8) * 255
            alpha = cropped_masked[:, :, 3:4] / 255.0
            cropped_rgb = (cropped_masked[:, :, :3] * alpha + white_bg * (1 - alpha)).astype(np.uint8)
        else:
            # No mask - just crop
            cropped_rgb = image_array[y1:y2, x1:x2]

        # Ensure minimum size
        if cropped_rgb.shape[0] < 10 or cropped_rgb.shape[1] < 10:
            return None

        # Convert to PIL Image for preprocessing
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(cropped_rgb)

        # Resize to 224x224
        pil_img = pil_img.resize((224, 224), PILImage.Resampling.BILINEAR)

        # To numpy CHW float32 normalized
        arr = np.asarray(pil_img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        chw = np.transpose(arr, (2, 0, 1))
        input_tensor = np.expand_dims(chw, axis=0).astype(np.float32)

        # Run ONNX inference
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_tensor})
        logits = outputs[0][0]

        # Softmax
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        probs = exp / np.sum(exp)

        # Get top prediction
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        top_class = FOOD_CLASS_NAMES[top_idx] if top_idx < len(FOOD_CLASS_NAMES) else f"class_{top_idx}"

        return {
            'class': top_class,
            'confidence': top_prob,
            'index': top_idx
        }

    except Exception as e:
        logger.warning(f"[VIT] Classification failed for detection: {e}")
        return None

def segment_food(model, image_array, confidence_threshold):
    """SegFormer segmentation using wrapper - image_array is RGB from PIL"""
    result = model.predict(image_array, confidence_threshold)
    # Keep masks for volume calculation - they will be removed later before JSON response
    return result

def deduplicate_overlapping_segments(detections: List[Dict], overlap_threshold: float = 0.3) -> List[Dict]:
    """Remove duplicate segments that significantly overlap.

    Keeps the segment with higher confidence when overlap is detected.
    """
    if not detections:
        return []

    filtered: List[Dict] = []

    for det1 in detections:
        mask1 = det1.get('mask')
        if mask1 is None:
            filtered.append(det1)
            continue

        is_duplicate = False
        for i, det2 in enumerate(filtered):
            mask2 = det2.get('mask')
            if mask2 is None:
                continue

            intersection = np.sum((mask1 > 0) & (mask2 > 0))
            area1 = np.sum(mask1 > 0)
            overlap_ratio = (intersection / area1) if area1 > 0 else 0.0

            if overlap_ratio > overlap_threshold:
                if det1.get('confidence', 0.0) > det2.get('confidence', 0.0):
                    filtered[i] = det1
                is_duplicate = True
                break

        if not is_duplicate:
            filtered.append(det1)

    return filtered

def detect_utensils(session, image_array, conf_thres=0.25, nms_iou=0.3, min_box_px=5, promote_top_n=None):
    """YOLO ONNX utensil detection with letterbox + class-agnostic NMS.

    Supports outputs shaped [1, C, N] or [1, N, C] with C in {6,7+}.

    Args:
        session: ONNX session
        image_array: RGB image
        conf_thres: Confidence threshold for detections
        nms_iou: IoU threshold for NMS
        min_box_px: Minimum box size in pixels
        promote_top_n: Promote top N detections (unused, for compatibility)
    """
    def letterbox(img, new_shape=(640, 640)):
        h, w = img.shape[:2]
        r = min(new_shape[0] / h, new_shape[1] / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dh, dw = new_shape[0] - nh, new_shape[1] - nw
        top, bottom = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return padded, r, left, top

    def nms(boxes, scores, iou_thres=0.5):
        if not boxes:
            return []
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1e-6) * (y2 - y1 + 1e-6)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_thres)[0]
            order = order[inds + 1]
        return keep

    logger.info("[UTENSIL] Starting YOLO ONNX utensil detection")
    input_size = 640
    lb_img, r, pad_x, pad_y = letterbox(image_array, (input_size, input_size))
    normalized = lb_img.astype(np.float32) / 255.0
    input_tensor = np.transpose(normalized, (2, 0, 1))[None, ...].astype(np.float32)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    logger.info(f"[UTENSIL] Model input: {input_name}, output: {output_name}")

    raw = session.run(None, {input_name: input_tensor})[0]
    logger.info(f"[UTENSIL] Raw output shape: {raw.shape}")

    # Normalize to [N, C]
    if raw.ndim == 3:
        # YOLO outputs are typically [1, num_detections, num_features]
        # Features: [x, y, w, h, conf, class1, class2, ...]
        if raw.shape[1] > raw.shape[2]:
            # [1, 8400, 7] -> transpose to [8400, 7]
            preds = raw[0]
        else:
            # [1, 7, 8400] -> transpose to [8400, 7]
            preds = raw[0].transpose(1, 0)
    else:
        preds = raw.reshape(-1, raw.shape[-1])

    logger.info(f"[UTENSIL] Predictions shape: {preds.shape}")

    # Use the provided confidence threshold
    boxes_xyxy = []
    scores = []
    class_ids = []

    for det in preds:
        C = len(det)
        # YOLO v8/v11 format: [cx, cy, w, h, class0, class1, ...]
        # No separate objectness score - class scores directly indicate presence
        if C < 5:
            continue
        cx, cy, w, h = det[:4]

        # For modern YOLO (v8/v11): Features [4:] are class scores
        class_scores = det[4:]
        cid = int(np.argmax(class_scores))
        conf = float(class_scores[cid])

        if conf < conf_thres:
            continue

        # If coordinates are normalized (<= 2.0), scale to letterboxed pixels
        if max(w, h, cx, cy) <= 2.0:
            cx *= input_size; cy *= input_size; w *= input_size; h *= input_size
        # Convert to xyxy in letterboxed image space
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        # Undo letterbox to original image coords
        # 1) remove padding
        x1 = x1 - pad_x
        y1 = y1 - pad_y
        x2 = x2 - pad_x
        y2 = y2 - pad_y
        # 2) scale back
        x1 /= r; y1 /= r; x2 /= r; y2 /= r
        h0, w0 = image_array.shape[:2]
        x1 = float(np.clip(x1, 0, w0 - 1)); y1 = float(np.clip(y1, 0, h0 - 1))
        x2 = float(np.clip(x2, 0, w0 - 1)); y2 = float(np.clip(y2, 0, h0 - 1))
        if (x2 - x1) >= min_box_px and (y2 - y1) >= min_box_px:
            boxes_xyxy.append([x1, y1, x2, y2])
            scores.append(conf)
            class_ids.append(cid)

    logger.info(f"[UTENSIL] Pre-NMS detections: {len(boxes_xyxy)}")

    keep = nms(boxes_xyxy, scores, iou_thres=nms_iou)
    out = []
    for i in keep:
        x1, y1, x2, y2 = boxes_xyxy[i]
        w_px = int(x2 - x1)
        h_px = int(y2 - y1)
        cid = class_ids[i]
        # Map YOLO class id to utensil name if known
        if 0 <= cid < len(YOLO_UTENSIL_LABELS):
            cname = YOLO_UTENSIL_LABELS[cid]
        else:
            cname = UTENSIL_SIZES.get(cid, {}).get("name", f"class_{cid}")
        detection = {
            "class_id": cid,
            "class_name": cname,
            "confidence": float(scores[i]),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "width_px": w_px,
            "height_px": h_px
        }
        out.append(detection)
        logger.info(f"[UTENSIL] Detected: {detection['class_name']} @ {detection['confidence']:.2f}")

    logger.info(f"[UTENSIL] Final detections: {len(out)}")
    return out

def calculate_pixel_to_mm_ratio(utensil_detections, spoon_len_mm=150.0, fork_len_mm=180.0, utensil_conf_threshold=0.5):
    """Calculate mm per pixel from utensil detections

    Args:
        utensil_detections: List of utensil detections
        spoon_len_mm: Real-world spoon length in mm
        fork_len_mm: Real-world fork length in mm
    """
    if not utensil_detections:
        return None

    # Normalize hybrid detections that may not have class_id
    rev_map = {v['name']: k for k, v in UTENSIL_SIZES.items()}
    norm = []
    for u in utensil_detections:
        u = dict(u)
        if 'class_id' not in u:
            label = u.get('class_name') or u.get('name') or u.get('type')
            if label in rev_map:
                u['class_id'] = rev_map[label]
                u['class_name'] = label
        # ensure width/height
        if 'width_px' not in u or 'height_px' not in u:
            if 'bbox' in u and isinstance(u['bbox'], (list, tuple)) and len(u['bbox']) == 4:
                x1, y1, x2, y2 = u['bbox']
                u['width_px'] = int(max(0, x2 - x1))
                u['height_px'] = int(max(0, y2 - y1))
        norm.append(u)
    utensil_detections = norm

    # Filter for confidence > 0.5 (aligned with non-API), then use highest confidence utensil
    valid_utensils = [u for u in utensil_detections if u['confidence'] >= utensil_conf_threshold]

    if not valid_utensils:
        return None

    # Use the highest confidence utensil
    best_utensil = max(valid_utensils, key=lambda x: x['confidence'])

    # Determine real-world length based on class name
    class_name = best_utensil.get('class_name', '').lower()
    real_length_mm = None

    if 'spoon' in class_name:
        real_length_mm = spoon_len_mm
    elif 'fork' in class_name:
        real_length_mm = fork_len_mm
    else:
        # Fallback to UTENSIL_SIZES if available
        class_id = best_utensil.get('class_id')
        if class_id in UTENSIL_SIZES:
            real_length_mm = UTENSIL_SIZES[class_id]['length_mm']

    if real_length_mm is not None:
        pixel_length = max(best_utensil['width_px'], best_utensil['height_px'])
        logger.info(f"[UTENSIL-DEBUG] Best: {best_utensil['class_name']} conf={best_utensil['confidence']:.2f}, real_len={real_length_mm}mm, pix_len={pixel_length}px")
        if pixel_length > 15:  # Lower minimum from 20 to 15 pixels
            ratio = real_length_mm / pixel_length
            logger.info(f"[UTENSIL-DEBUG] Ratio={ratio:.3f} mm/px, sanity check: {0.05 <= ratio <= 3.0}")
            # Wider sanity check range: 0.05 to 3.0 mm/px
            if 0.05 <= ratio <= 3.0:
                logger.info(f"[UTENSIL] Using {best_utensil['class_name']} for scaling: {ratio:.3f} mm/px (conf: {best_utensil['confidence']:.2f})")
                return ratio
        else:
            logger.info(f"[UTENSIL-DEBUG] Pixel length {pixel_length} <= 15, rejected")
    else:
        logger.info(f"[UTENSIL-DEBUG] No real_length_mm found for {best_utensil.get('class_name')}")

    return None

def synthetic_pixel_to_mm_ratio(image_array, px_per_mm_1024=3.5):
    """Calculate synthetic pixel-to-mm ratio based on image width

    Args:
        image_array: Image array
        px_per_mm_1024: Pixels per mm at 1024px width (calibrated value)

    Returns:
        mm per pixel ratio
    """
    NORMALIZED_WIDTH = 1024
    scale_to_original = image_array.shape[1] / NORMALIZED_WIDTH
    pixels_per_mm = px_per_mm_1024 * scale_to_original
    mm_per_pixel = 1.0 / pixels_per_mm
    return mm_per_pixel

def calculate_volumes(image_array, detections, pixel_to_mm_ratio, depth_model):
    """UNUSED: Legacy ellipsoid method - kept for reference only.

    This function is no longer used. The API always uses depth_map integration
    with food_95th baseline via calculate_volumes_depth_map().
    """
    volumes = {}

    logger.info(f"[VOL] Starting volume calculation for {len(detections)} detections")
    logger.info(f"[VOL] Pixel to mm ratio: {pixel_to_mm_ratio}")

    # Convert mm to cm for calculations
    if pixel_to_mm_ratio is None:
        # Calibrated fallback identical to non_api_working:
        # 3.5 pixels per mm at width=1024; scale by image width
        NORMALIZED_WIDTH = 1024
        TYPICAL_PIXELS_PER_MM_AT_1024 = 3.5
        scale_to_original = image_array.shape[1] / NORMALIZED_WIDTH
        pixels_per_mm = TYPICAL_PIXELS_PER_MM_AT_1024 * scale_to_original
        pixel_to_cm = (1.0 / pixels_per_mm) / 10.0  # mm->cm
        logger.info(f"[VOL] Using fallback pixel_to_cm={pixel_to_cm:.4f} via 3.5 px/mm calibration")
    else:
        pixel_to_cm = pixel_to_mm_ratio / 10.0  # mm->cm
        logger.info(f"[VOL] Using utensil-based pixel_to_cm={pixel_to_cm:.4f}")

    # Sanity check bounds; fallback to calibrated default if out of range
    if pixel_to_cm < 0.01 or pixel_to_cm > 0.5:
        pixels_per_mm = 3.5 * (image_array.shape[1] / 1024)
        pixel_to_cm = (1.0 / pixels_per_mm) / 10.0
        logger.warning(f"[VOL] pixel_to_cm out of range, reverting to calibrated default {pixel_to_cm:.4f}")

    for i, detection in enumerate(detections):
        # Get mask from detection (added by segmentation_wrapper)
        mask = detection.get('mask')
        logger.info(f"[VOL] Detection {i} ({detection['class_name']}): mask={'present' if mask is not None else 'MISSING'}")

        if mask is None:
            logger.warning(f"[VOL] Detection {i}: SKIPPED (no mask in detection dict)")
            continue

        if not np.any(mask):
            logger.warning(f"[VOL] Detection {i}: SKIPPED (mask is all zeros)")
            continue

        # Use regionprops for accurate measurements (like original)
        labeled_mask = (mask > 0).astype(int)
        props = regionprops(labeled_mask)
        if not props:
            logger.warning(f"[VOL] Detection {i}: SKIPPED (regionprops empty)")
            continue

        prop = props[0]

        # Calculate volume using EXACT ellipsoid method from non_api_working
        # Get dimensions in pixels from BBOX (not major/minor axis)
        height_px = prop.bbox[2] - prop.bbox[0]
        width_px = prop.bbox[3] - prop.bbox[1]

        # Convert to cm
        height_cm = height_px * pixel_to_cm
        width_cm = width_px * pixel_to_cm

        # Estimate depth (assume roughly circular cross-section, depth = 60% of width)
        depth_cm = width_cm * 0.6

        # Calculate ellipsoid volume: (4/3) * π * a * b * c
        volume_cm3 = (4/3) * math.pi * (width_cm/2) * (height_cm/2) * (depth_cm/2)

        # Account for food compactness (foods are not perfect ellipsoids)
        # This automatically reduces volume for sparse/scattered detections
        bbox_area = (prop.bbox[2] - prop.bbox[0]) * (prop.bbox[3] - prop.bbox[1])
        compactness_factor = prop.filled_area / bbox_area if bbox_area > 0 else 1.0

        volume_cm3 *= compactness_factor

        # Convert cm³ to ml (1:1 ratio)
        volume_ml = volume_cm3

        # Sanity checks to prevent unrealistic values
        if volume_ml < 5.0:
            logger.info(f"[VOL] Raising minimum volume from {volume_ml:.1f}ml to 5ml for {detection['class_name']}")
            volume_ml = 5.0
        elif volume_ml > 800.0:
            # Cap at 800 ml to align with working API sanity bound
            logger.warning(f"[VOL] Capping excessive volume from {volume_ml:.1f}ml to 800ml for {detection['class_name']} (compactness={compactness_factor:.2f}, area={prop.filled_area}px)")
            volume_ml = 800.0

        volumes[f"detection_{i}"] = {
            "volume_ml": float(volume_ml),
            "area_cm2": float(prop.filled_area * (pixel_to_cm ** 2)),
            "estimated_depth_cm": float(depth_cm),
            "class_name": detection['class_name'],
            "compactness_factor": float(compactness_factor),
            "dimensions_cm": {
                "width": float(width_cm),
                "height": float(height_cm),
                "depth": float(depth_cm)
            }
        }

        logger.info(f"[VOL] {detection['class_name']}: {volume_ml:.1f}ml (w={width_cm:.1f}cm, h={height_cm:.1f}cm, d={depth_cm:.1f}cm, compact={compactness_factor:.2f})")

    # Calculate totals
    total_volume_ml = sum(v['volume_ml'] for v in volumes.values())
    total_area_cm2 = sum(v['area_cm2'] for v in volumes.values())

    # IMPORTANT: Cap total volume for a single meal to prevent over-estimation
    # Multiple overlapping detections can inflate volume estimates
    MAX_MEAL_VOLUME_ML = 800.0
    if total_volume_ml > MAX_MEAL_VOLUME_ML:
        logger.warning(f"[VOL] Total volume {total_volume_ml:.1f}ml exceeds reasonable meal size, capping to {MAX_MEAL_VOLUME_ML}ml")
        # Scale down all individual volumes proportionally
        scale_factor = MAX_MEAL_VOLUME_ML / total_volume_ml
        for key in list(volumes.keys()):
            if key.startswith('detection_'):
                volumes[key]['volume_ml'] = float(volumes[key]['volume_ml'] * scale_factor)
        total_volume_ml = MAX_MEAL_VOLUME_ML

    # Add totals to the dict
    volumes['total_volume_ml'] = float(total_volume_ml)
    volumes['total_area_cm2'] = float(total_area_cm2)

    logger.info(f"[VOL] Calculated volumes for {len(volumes)-2}/{len(detections)} detections, TOTAL: {total_volume_ml:.1f}ml")
    return volumes

def compute_depth_map(depth_session: ort.InferenceSession, image_array: np.ndarray) -> np.ndarray:
    """Run ONNX depth model to produce a depth map resized to image size.

    This reads model input shape dynamically and applies simple [0,1] normalization.
    """
    h, w = image_array.shape[:2]
    # Determine input
    input_info = depth_session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape  # [N, C, H, W] possibly with symbolic dims
    # DepthAnything (ViT-14) requires H and W divisible by 14. Use 392x392 if dynamic.
    def _to_int(v, default):
        try:
            return int(v)
        except Exception:
            return default
    target_h = _to_int(input_shape[2] if len(input_shape) > 2 else None, 392)
    target_w = _to_int(input_shape[3] if len(input_shape) > 3 else None, 392)
    # Enforce multiples of 14
    patch = 14
    if target_h % patch != 0:
        target_h = int(np.ceil(target_h / patch) * patch)
    if target_w % patch != 0:
        target_w = int(np.ceil(target_w / patch) * patch)

    # Preprocess (RGB -> float32, [0,1], CHW)
    resized = cv2.resize(image_array, (target_w, target_h))
    img = resized.astype(np.float32) / 255.0
    chw = np.transpose(img, (2, 0, 1))[None, ...].astype(np.float32)

    # Inference
    depth_pred = depth_session.run(None, {input_name: chw})[0]
    # Squeeze to HxW (handle [1,1,H,W] or [1,H,W])
    depth_map = depth_pred.squeeze()
    if depth_map.ndim == 3:
        # If CxHxW, take first channel
        depth_map = depth_map[0]

    # Resize back to original image size
    depth_map = cv2.resize(depth_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize to [0,1] for stability
    dmin, dmax = float(depth_map.min()), float(depth_map.max())
    if dmax - dmin > 1e-6:
        depth_map = (depth_map - dmin) / (dmax - dmin)
    else:
        depth_map = np.zeros_like(depth_map, dtype=np.float32)

    return depth_map

def visualize_depth_map(depth_map: np.ndarray) -> np.ndarray:
    """Convert a normalized [0,1] depth map to an RGB visualization with contrast stretch.

    Depth Anything outputs inverse depth (higher = farther). We invert so darker ~ farther,
    lighter ~ closer, and stretch contrast using 2nd-98th percentile.
    """
    dm = depth_map.astype(np.float32)
    # Robust contrast stretching
    p2, p98 = np.percentile(dm, [2, 98])
    if p98 - p2 > 1e-6:
        dm = (dm - p2) / (p98 - p2)
    else:
        dm = (dm - dm.min()) / (dm.max() - dm.min() + 1e-6)
    dm = np.clip(dm, 0.0, 1.0)
    # Invert so nearer is brighter
    dm = 1.0 - dm
    dm8 = (dm * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(dm8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

_CLASS_COLORS = {
    'fruit': (255, 0, 0),
    'vegetable': (0, 200, 0),
    'carbohydrate': (0, 120, 255),
    'protein': (255, 0, 200),
    'dairy': (255, 215, 0),
    'fat': (0, 255, 255),
    'other': (255, 255, 255),
}

def create_segmentation_overlay(image_array: np.ndarray, detections: List[Dict], utensils: List[Dict]) -> np.ndarray:
    """Create an RGB overlay image with semi-transparent masks and bounding boxes."""
    img = image_array.copy()
    overlay = img.copy()
    h, w = img.shape[:2]
    # Blend masks
    for det in detections:
        mask = det.get('mask')
        if mask is None or not np.any(mask):
            continue
        cls = det.get('class_name', 'other')
        color = _CLASS_COLORS.get(cls, (255, 255, 255))
        # Create color layer
        color_layer = np.zeros_like(img, dtype=np.uint8)
        color_layer[mask > 0] = color
        # Alpha blend
        overlay = cv2.addWeighted(overlay, 1.0, color_layer, 0.35, 0)

    # Draw detection boxes
    for det in detections:
        x1, y1, x2, y2 = det.get('bbox', [0, 0, 0, 0])
        cls = det.get('class_name', 'item')
        color = _CLASS_COLORS.get(cls, (255, 255, 255))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

    # Draw utensil boxes in yellow
    for u in utensils or []:
        bbox = u.get('bbox')
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 215, 0), 2)

    return overlay

def encode_png_base64(img_rgb: np.ndarray) -> str:
    """Encode an RGB image array as data:image/png;base64 string."""
    if img_rgb.dtype != np.uint8:
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_rgb)
    buf = BytesIO()
    pil_img.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('utf-8')

def detect_utensil_candidates_debug(session: ort.InferenceSession, image_array: np.ndarray, top_k: int = 20) -> List[Dict]:
    """Return top-K utensil candidates (pre-threshold) for debugging overlays."""
    try:
        input_size = 640
        # Letterbox
        h, w = image_array.shape[:2]
        r = min(input_size / h, input_size / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(image_array, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dh, dw = input_size - nh, input_size - nw
        top, bottom = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2
        lb_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        # Normalize & NCHW
        inp = (lb_img.astype(np.float32) / 255.0)
        inp = np.transpose(inp, (2, 0, 1))[None, ...]
        raw = session.run(None, {session.get_inputs()[0].name: inp})[0]
        # shape handling
        if raw.ndim == 3:
            if raw.shape[1] > raw.shape[2]:
                preds = raw[0]
            else:
                preds = raw[0].transpose(1, 0)
        else:
            preds = raw.reshape(-1, raw.shape[-1])
        # Build candidates
        cands = []
        for det in preds:
            if len(det) < 5:
                continue
            cx, cy, bw, bh = det[:4]
            # YOLO v8/v11: features [4:] are class scores (no separate objectness)
            class_scores = det[4:]
            cid = int(np.argmax(class_scores))
            conf = float(class_scores[cid])
            # If outputs are normalized, scale up to 640 letterbox space
            if max(bw, bh, cx, cy) <= 2.0:
                cx *= input_size; cy *= input_size; bw *= input_size; bh *= input_size
            # Undo letterbox
            x1 = (cx - bw/2) - left; y1 = (cy - bh/2) - top
            x2 = (cx + bw/2) - left; y2 = (cy + bh/2) - top
            x1 /= r; y1 /= r; x2 /= r; y2 /= r
            x1 = float(np.clip(x1, 0, w-1)); y1 = float(np.clip(y1, 0, h-1))
            x2 = float(np.clip(x2, 0, w-1)); y2 = float(np.clip(y2, 0, h-1))
            if (x2 - x1) < 5 or (y2 - y1) < 5:
                continue
            cands.append({
                'class_id': cid,
                'confidence': conf,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
            })
        # Sort by confidence and cap
        cands = sorted(cands, key=lambda d: d['confidence'], reverse=True)[:top_k]
        return cands
    except Exception as e:
        logger.warning(f"[UTENSIL-DEBUG] Failed: {e}")
        return []

def calculate_volumes_depth_map(
    image_array: np.ndarray,
    detections: List[Dict],
    pixel_to_mm_ratio: float | None,
    depth_map: np.ndarray,
    baseline_method: str = 'food_95th',
    depth_upscale_factor: float = 1.0,
    vit_classifications: Dict[int, Dict] = None,
    classification_confidence_threshold: float = 0.40,
) -> Dict[str, Dict]:
    """Calculate volumes using depth-map integration aligned with non_api_working (95th baseline).

    Mirrors non_api_working's calculate_food_volume_depth_map:
    - Baseline from mask region (default 95th percentile, i.e., furthest plate surface)
    - Relative heights = baseline_depth - food_depths (clipped >=0)
    - Height scaling via diameter-based heuristic with variance thresholds
    - Depth scale clamped to [30, 300] cm per unit depth range
    - Weight/volume outlier correction to match non-API consistency
    """
    volumes: Dict[str, Dict] = {}
    vit_classifications = vit_classifications or {}

    # Calculate total food area for outlier detection
    img_height, img_width = image_array.shape[:2]
    img_total_pixels = img_height * img_width
    total_food_area = sum(np.sum(det.get('mask', np.zeros((img_height, img_width))) > 0) for det in detections)
    food_coverage_pct = (total_food_area / img_total_pixels) * 100.0

    # Pixel to cm conversion
    if pixel_to_mm_ratio is None:
        NORMALIZED_WIDTH = 1024
        TYPICAL_PIXELS_PER_MM_AT_1024 = 3.5
        scale_to_original = image_array.shape[1] / NORMALIZED_WIDTH
        pixels_per_mm = TYPICAL_PIXELS_PER_MM_AT_1024 * scale_to_original
        pixel_to_cm = (1.0 / pixels_per_mm) / 10.0
    else:
        pixel_to_cm = pixel_to_mm_ratio / 10.0

    if pixel_to_cm < 0.01 or pixel_to_cm > 0.5:
        pixels_per_mm = 3.5 * (image_array.shape[1] / 1024)
        pixel_to_cm = (1.0 / pixels_per_mm) / 10.0

    for i, det in enumerate(detections):
        mask = det.get('mask')
        if mask is None or not np.any(mask):
            continue

        # Ensure mask matches depth map dimensions if needed
        if depth_map.shape[:2] != mask.shape[:2]:
            original_mask_h = mask.shape[0]
            mask = cv2.resize(mask.astype(np.uint8), (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_NEAREST)
            actual_scale_factor = depth_map.shape[0] / original_mask_h if original_mask_h > 0 else 1.0
            pixel_to_cm_adjusted = pixel_to_cm / actual_scale_factor
        else:
            pixel_to_cm_adjusted = pixel_to_cm

        food_pixels = mask > 0
        food_depths = depth_map[food_pixels]
        if food_depths.size == 0:
            continue

        # Calculate baseline depth based on selected method
        if baseline_method == "food_5th":
            # 5th percentile: nearest food surface (good for bowls/containers)
            baseline_depth = float(np.percentile(food_depths, 5))
            baseline_source = "food_5th (5th percentile)"
        elif baseline_method == "adaptive":
            # Adaptive: choose based on depth variance
            depth_var = float(np.var(food_depths))
            if depth_var > 0.01:  # High variance suggests bowl
                baseline_depth = float(np.percentile(food_depths, 5))
                baseline_source = "adaptive (5th percentile, high variance)"
            else:  # Low variance suggests flat plate
                baseline_depth = float(np.percentile(food_depths, 95))
                baseline_source = "adaptive (95th percentile, low variance)"
        else:  # Default: food_95th
            # 95th percentile: furthest plate surface (best for flat plates)
            baseline_depth = float(np.percentile(food_depths, 95))
            baseline_source = "food_95th (95th percentile)"

        # Relative heights (Depth Anything inverse depth: higher=farther)
        relative_heights = baseline_depth - food_depths
        relative_heights = np.maximum(0.0, relative_heights)

        # Diameter-based height scaling
        area_px = float(np.sum(food_pixels))
        equiv_diam_px = 2.0 * math.sqrt(area_px / math.pi)
        equiv_diam_cm = equiv_diam_px * pixel_to_cm_adjusted
        depth_range = float(np.ptp(relative_heights))

        if depth_range > 1e-4:
            var = float(np.var(relative_heights))
            high_var_threshold = 0.005
            low_var_threshold = 0.001
            if var > high_var_threshold:
                ratio = 0.25
            elif var < low_var_threshold:
                ratio = 0.12
            else:
                ratio = 0.18
            est_height_cm = float(np.clip(equiv_diam_cm * ratio, 0.5, 15.0))
            depth_scale_cm = est_height_cm / max(depth_range, 1e-6)
            depth_scale_cm = float(np.clip(depth_scale_cm, 30.0, 300.0))
        else:
            depth_scale_cm = 1.0

        rel_heights_cm = relative_heights * depth_scale_cm
        pixel_area_cm2 = pixel_to_cm_adjusted ** 2
        volume_cm3 = float(np.sum(rel_heights_cm) * pixel_area_cm2)
        volume_ml = float(volume_cm3)

        # Get per-detection classification info
        food_type = det.get('class_name', 'unknown')  # From segmentation (broad class)
        specific_class = None
        specific_confidence = 0.0
        # Prefer provided VIT classifications dict if available
        if i in vit_classifications:
            vit_result = vit_classifications[i]
            specific_class = vit_result.get('class')
            specific_confidence = vit_result.get('confidence', 0.0)
        else:
            # Fallback to detection-level fields populated earlier in the pipeline
            specific_class = det.get('specific_food')
            specific_confidence = det.get('specific_confidence', 0.0)

        # Get density for weight calculation
        nutrition_db = MODELS.get('nutrition_db')
        if nutrition_db:
            # Use specific density only when classification is confident; otherwise broad
            use_specific = bool(specific_class) and (specific_confidence >= classification_confidence_threshold)
            if use_specific:
                nutrition_info = nutrition_db.get_nutrition_info(specific_class)
                if nutrition_info is not None:
                    density_used = nutrition_info.density_g_ml
                    density_source = f"Specific ({specific_class})"
                else:
                    # If specific not found, fall back to broad
                    nutrition_info = nutrition_db.get_nutrition_info(food_type)
                    density_used = nutrition_info.density_g_ml if nutrition_info else 0.8
                    density_source = f"Broad ({food_type})"
            else:
                # Use broad class density
                nutrition_info = nutrition_db.get_nutrition_info(food_type)
                density_used = nutrition_info.density_g_ml if nutrition_info else 0.8
                density_source = f"Broad ({food_type})"
        else:
            # Fallback density
            density_used = 0.8
            density_source = "Default"

        # Calculate weight from volume
        weight_grams = volume_ml * density_used

        # Enhanced outlier detection with multiple sanity checks (CUMULATIVE)
        # Matches non_api_working logic (lines 1321-1353)
        is_outlier = False

        # Rule 0: Depth map tends to overestimate - apply 0.6x correction preemptively
        weight_grams = weight_grams * 0.6
        volume_ml = volume_ml * 0.6
        is_outlier = True

        # Rule 1: Small area, high weight
        if food_coverage_pct < 25 and weight_grams > 400:
            weight_grams = weight_grams * 0.5
            volume_ml = volume_ml * 0.5
            is_outlier = True

        # Rule 2: Single segment weight > 300g
        if weight_grams > 300:
            weight_grams = weight_grams * 0.6
            volume_ml = volume_ml * 0.6
            is_outlier = True

        # Rule 3: Unrealistic density check (actual vs expected)
        if volume_ml > 0:
            actual_density = weight_grams / volume_ml
            # If actual density is > 2x expected density
            if actual_density > density_used * 2:
                correction = density_used / actual_density
                weight_grams = weight_grams * correction
                volume_ml = volume_ml * correction
                is_outlier = True

        volumes[f"detection_{i}"] = {
            "volume_ml": float(volume_ml),
            "area_cm2": float(area_px * pixel_area_cm2),
            "estimated_depth_cm": float(np.max(rel_heights_cm) if rel_heights_cm.size else 0.0),
            "class_name": food_type,
            "specific_class": specific_class,
            "specific_confidence": specific_confidence,
            "weight_grams": float(weight_grams),
            "density_g_ml": float(density_used),
            "density_source": density_source,
            "baseline_depth": baseline_depth,
            "outlier_corrected": is_outlier,
        }

    # Totals
    total_volume_ml = sum(v['volume_ml'] for v in volumes.values())
    total_area_cm2 = sum(v['area_cm2'] for v in volumes.values())
    volumes['total_volume_ml'] = float(total_volume_ml)
    volumes['total_area_cm2'] = float(total_area_cm2)
    return volumes

def calculate_weighted_classification(
    image_array: np.ndarray,
    detections: List[Dict],
    full_image_prediction: Dict,
    min_weighted_score_pct: float = 2.0
) -> Dict:
    """Calculate area-weighted classification from segments.

    Implements the sophisticated weighted segment analysis from non_api_working:
    - Weighted score = confidence² × area_fraction × confidence_boost × size_penalty
    - Multi-segment bonus for images with multiple substantial food items
    - Filters segments below minimum weighted score threshold

    Args:
        image_array: RGB image array
        detections: List of detection dicts with 'specific_food', 'specific_confidence', 'mask'
        full_image_prediction: Full image classification result
        min_weighted_score_pct: Minimum weighted score percentage to include segment (default 2.0%)

    Returns:
        Dict with:
            - weighted_confidence: float
            - best_segment: dict or None
            - all_segments: list of segment contributions
            - multi_segment_bonus: float
            - recommendation: "full_image" | "weighted_segments" | "similar"
    """
    if not detections:
        return {
            "weighted_confidence": 0.0,
            "best_segment": None,
            "all_segments": [],
            "multi_segment_bonus": 1.0,
            "recommendation": "full_image"
        }

    # Get full image confidence as reference
    reference_confidence = full_image_prediction.get('confidence', 0.0)

    # Calculate total segmented area (only valid food segments with VIT classification)
    h, w = image_array.shape[:2]
    total_segmented_area = 0
    valid_segments = []

    for det in detections:
        specific_confidence = det.get('specific_confidence', 0.0)
        if specific_confidence > 0:
            mask = det.get('mask')
            if mask is not None:
                # Resize mask to match image if needed
                if mask.shape[:2] != (h, w):
                    mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                else:
                    mask_resized = mask.astype(np.uint8)

                mask_area = np.sum(mask_resized > 0)
                total_segmented_area += mask_area

                valid_segments.append({
                    'specific_class': det.get('specific_food'),
                    'specific_confidence': specific_confidence,
                    'mask_area': mask_area,
                    'broad_class': det.get('class_name', 'unknown')
                })

    if not valid_segments or total_segmented_area == 0:
        return {
            "weighted_confidence": 0.0,
            "best_segment": None,
            "all_segments": [],
            "multi_segment_bonus": 1.0,
            "recommendation": "full_image"
        }

    # Filter segments by minimum weighted score (before applying bonuses)
    filtered_segments = []
    for seg in valid_segments:
        area_fraction = seg['mask_area'] / total_segmented_area
        weighted_score_pct = (seg['specific_confidence'] * area_fraction) * 100
        if weighted_score_pct >= min_weighted_score_pct:
            filtered_segments.append(seg)

    if not filtered_segments:
        return {
            "weighted_confidence": 0.0,
            "best_segment": None,
            "all_segments": [],
            "multi_segment_bonus": 1.0,
            "recommendation": "full_image"
        }

    # Count big segments (> 10% of total food area)
    big_segment_threshold = 0.10
    num_big_segments = sum(
        1 for seg in filtered_segments
        if (seg['mask_area'] / total_segmented_area) > big_segment_threshold
    )

    # Multi-segment bonus (rewards having multiple substantial food items)
    multi_segment_bonus = 1.0 + max(0, num_big_segments - 1) * 0.30

    # Calculate weighted confidence with all factors
    # confidence² × area_fraction × confidence_boost × size_penalty
    segment_scores = []
    for seg in filtered_segments:
        area_fraction = seg['mask_area'] / total_segmented_area
        confidence = seg['specific_confidence']

        # Confidence boost: segments more confident than full image get boosted
        confidence_boost = max(1.0, confidence / reference_confidence if reference_confidence > 0 else 1.0)

        # Size penalty: penalize segments < 5% of total food area
        size_penalty = min(1.0, area_fraction / 0.05)

        # Calculate weighted score
        weighted_score = (confidence ** 2) * area_fraction * confidence_boost * size_penalty

        segment_scores.append({
            'segment': seg,
            'weighted_score': weighted_score,
            'weighted_score_pct': weighted_score * 100
        })

    # Apply multi-segment bonus
    raw_weighted_confidence = sum(s['weighted_score'] for s in segment_scores) * multi_segment_bonus

    # Normalize to ensure max is 100% (1.0)
    weighted_confidence = min(1.0, raw_weighted_confidence)

    # Find best segment (highest weighted score)
    best_segment_info = max(segment_scores, key=lambda x: x['weighted_score'])
    best_segment = best_segment_info['segment']

    # Sort all segments by weighted score (highest first)
    all_segments = sorted(segment_scores, key=lambda x: x['weighted_score'], reverse=True)

    # Determine recommendation
    if weighted_confidence > reference_confidence + 0.05:
        recommendation = "weighted_segments"
    elif reference_confidence > weighted_confidence + 0.05:
        recommendation = "full_image"
    else:
        recommendation = "similar"

    return {
        "weighted_confidence": float(weighted_confidence),
        "best_segment": {
            "class": best_segment['specific_class'],
            "confidence": float(best_segment['specific_confidence']),
            "broad_class": best_segment['broad_class']
        },
        "all_segments": [
            {
                "class": s['segment']['specific_class'],
                "confidence": float(s['segment']['specific_confidence']),
                "weighted_score_pct": float(s['weighted_score_pct'])
            }
            for s in all_segments
        ],
        "multi_segment_bonus": float(multi_segment_bonus),
        "num_big_segments": int(num_big_segments),
        "recommendation": recommendation
    }

def calculate_nutrition(detections, classification_result, volume_estimates, nutrition_db, classification_confidence_threshold=0.40):
    """Calculate nutrition using per-detection classification + volumes.

    Matches non_api_working logic (lines 2300-2376):
    - Uses per-detection specific_class from VIT if confidence >= threshold
    - Falls back to broad class from segmentation if confidence < threshold
    - Calculates nutrition from weight (already computed in volume_estimates)
    """
    nutrition_items = []
    total_calories = 0.0
    total_protein = 0.0
    total_carbs = 0.0
    total_fat = 0.0
    total_fiber = 0.0
    total_volume = 0.0
    total_weight = 0.0

    for i, detection in enumerate(detections):
        volume_key = f"detection_{i}"
        volume_info = volume_estimates.get(volume_key, {})

        if not volume_info:
            continue

        # Get per-detection classification info (stored by calculate_volumes_depth_map)
        specific_class = volume_info.get('specific_class')
        specific_confidence = volume_info.get('specific_confidence', 0.0)
        food_type = volume_info.get('class_name', detection.get('class_name', 'unknown'))
        weight_g = volume_info.get('weight_grams', 0.0)
        volume_ml = volume_info.get('volume_ml', 0.0)

        # Determine which classification to use based on confidence threshold
        use_specific = specific_class and specific_confidence >= classification_confidence_threshold

        if use_specific:
            # Use specific classification (confidence above threshold)
            nutrition_info = nutrition_db.get_nutrition_info(specific_class)
            display_name = specific_class.replace('_', ' ').title()
            confidence_display = f"{specific_confidence:.1%}"
        else:
            # Fall back to broad category (confidence below threshold or no specific class)
            nutrition_info = nutrition_db.get_nutrition_info(food_type)
            display_name = f"{food_type.replace('_', ' ').title()} (Broad)"
            confidence_display = f"{specific_confidence:.1%}*" if specific_class else "N/A"

        if nutrition_info:
            # Calculate nutrition based on weight (nutrition_info is per 100g)
            weight_ratio = weight_g / 100.0
            calories = nutrition_info.calories_per_100g * weight_ratio
            protein = nutrition_info.protein_g * weight_ratio
            carbs = nutrition_info.carbs_g * weight_ratio
            fat = nutrition_info.fat_g * weight_ratio
            fiber = nutrition_info.fiber_g * weight_ratio

            # Add to totals
            total_calories += calories
            total_protein += protein
            total_carbs += carbs
            total_fat += fat
            total_fiber += fiber
            total_volume += volume_ml
            total_weight += weight_g

            nutrition_items.append({
                "class_name": food_type,
                "specific_food": specific_class if use_specific else None,
                "display_name": display_name,
                "confidence": confidence_display,
                "volume_ml": volume_ml,
                "weight_g": weight_g,
                "calories": calories,
                "protein_g": protein,
                "carbs_g": carbs,
                "fat_g": fat,
                "fiber_g": fiber,
            })

    summary = {
        "food_name": classification_result["top_prediction"]["class_name"],
        "total_calories": total_calories,
        "total_volume_ml": total_volume,
        "total_weight_g": total_weight,
        "total_protein_g": total_protein,
        "total_carbs_g": total_carbs,
        "total_fat_g": total_fat,
        "total_fiber_g": total_fiber,
        "item_count": len(nutrition_items),
        "classification_threshold": classification_confidence_threshold,
    }

    return {
        "items": nutrition_items,
        "summary": summary
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )