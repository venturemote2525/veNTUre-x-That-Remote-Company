from PIL import Image
import numpy as np
import io

def preprocess_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Example: adjust to your training preprocessing
    image = image.resize((224, 224))  
    arr = np.array(image).astype(np.float32) / 255.0

    if arr.ndim == 2:  # grayscale → 3 channels
        arr = np.stack([arr] * 3, axis=-1)

    # HWC → CHW
    arr = np.transpose(arr, (2, 0, 1))
    # Add batch dimension
    arr = np.expand_dims(arr, axis=0)

    return arr