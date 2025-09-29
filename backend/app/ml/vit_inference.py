# app/ml/vit_inference.py
from pathlib import Path
from PIL import Image
import onnxruntime as ort
import numpy as np
import os
from typing import Dict, Any, List, Tuple
from app.ml.class_names import FOOD_CLASS_NAMES

class FoodClassificationAPI:
    def __init__(self, model_path: str = "app/ai_models/vit_small.onnx", providers: List[str] = None):
        self.model_path = model_path
        self.providers = providers or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.model_path, providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB").resize((224, 224))
        arr = np.array(image).astype("float32") / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, axis=0)
        return arr

    def run_inference(self, arr: np.ndarray) -> np.ndarray:
        return self.session.run([self.output_name], {self.input_name: arr})[0]

    def postprocess(self, outputs: np.ndarray, top_k: int = 5):
        probs = self._softmax(outputs[0])
        top_idx = np.argsort(probs)[::-1][:top_k]
        return [(FOOD_CLASS_NAMES[i], float(probs[i])) for i in top_idx]

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def analyze(self, image: Image.Image, top_k: int = 5) -> Dict[str, Any]:
        arr = self.preprocess_image(image)
        outputs = self.run_inference(arr)
        preds = self.postprocess(outputs, top_k)
        return {
            "food_name": preds[0][0],
            "confidence": preds[0][1],
            # ⚡️ Hook up with nutrition DB later:
            "calories": 650,
            "protein": 25,
            "carbs": 45,
            "fat": 15,
        }

# Global instance so model is loaded once
classifier = FoodClassificationAPI()

def analyze_food_image(image: Image.Image, top_k: int = 5) -> Dict[str, Any]:
    return classifier.analyze(image, top_k)
