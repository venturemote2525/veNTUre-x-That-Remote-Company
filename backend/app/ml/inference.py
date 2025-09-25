import onnxruntime as ort
import numpy as np
from app.ml.preprocess import preprocess_image

class ONNXClassifier:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, image_bytes: bytes):
        # Preprocess raw image bytes
        arr = preprocess_image(image_bytes)

        # Run inference
        preds = self.session.run([self.output_name], {self.input_name: arr})[0]

        pred_idx = int(np.argmax(preds))
        confidence = float(np.max(preds))

        return {"class_id": pred_idx, "confidence": confidence}
    
