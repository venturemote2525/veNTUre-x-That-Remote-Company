import onnxruntime as ort
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import os
from pathlib import Path

# Get the path to the model
MODEL_PATH = Path(__file__).parent.parent / "ai_models" / "vit_small.onnx"

class VitInference:
    def __init__(self):
        self.session = None
        self.transform = None
        self._load_model()
        self._setup_transforms()

    def _load_model(self):
        """Load the ONNX model"""
        try:
            self.session = ort.InferenceSession(str(MODEL_PATH))
            print(f"Successfully loaded ViT model from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def _setup_transforms(self):
        """Setup image preprocessing transforms for ViT"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ViT typically uses 224x224 images
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]   # ImageNet stds
            )
        ])

    def preprocess_image(self, image: Image.Image):
        """Preprocess PIL Image for ViT model"""
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transforms
        tensor = self.transform(image)

        # Add batch dimension and convert to numpy
        batch_tensor = tensor.unsqueeze(0)
        return batch_tensor.numpy()

    def predict(self, image: Image.Image):
        """Run inference on the image"""
        try:
            # Preprocess image
            input_array = self.preprocess_image(image)

            # Get input name from the model
            input_name = self.session.get_inputs()[0].name

            # Run inference
            outputs = self.session.run(None, {input_name: input_array})

            # Process outputs - this depends on your model's output format
            # For now, returning dummy food classification results
            predictions = outputs[0][0]  # Assuming first output, first batch item

            # Convert to food classification results
            # You'll need to map your model's outputs to food categories and nutritional info
            return self._process_predictions(predictions)

        except Exception as e:
            print(f"Error during inference: {e}")
            # Return dummy data if inference fails
            return {
                "food_name": "Unknown Food",
                "calories": 300,
                "protein": 15,
                "carbs": 30,
                "fat": 10,
                "confidence": 0.5
            }

    def _process_predictions(self, predictions):
        """Process model predictions into food nutrition data"""
        # This is where you'd implement the logic to convert
        # your ViT model's outputs into food nutrition information

        # For now, returning sample data based on predictions
        # You'll need to customize this based on your model's actual outputs

        max_idx = np.argmax(predictions)
        confidence = float(np.max(predictions))

        # Sample food mapping - replace with your actual mapping
        food_mapping = {
            0: {"name": "Apple", "calories": 95, "protein": 0, "carbs": 25, "fat": 0},
            1: {"name": "Banana", "calories": 105, "protein": 1, "carbs": 27, "fat": 0},
            2: {"name": "Chicken Breast", "calories": 230, "protein": 43, "carbs": 0, "fat": 5},
            3: {"name": "Rice Bowl", "calories": 350, "protein": 8, "carbs": 70, "fat": 2},
            # Add more mappings based on your model's classes
        }

        food_info = food_mapping.get(max_idx % len(food_mapping), {
            "name": "Mixed Dish",
            "calories": 400,
            "protein": 20,
            "carbs": 40,
            "fat": 15
        })

        return {
            "food_name": food_info["name"],
            "calories": food_info["calories"],
            "protein": food_info["protein"],
            "carbs": food_info["carbs"],
            "fat": food_info["fat"],
            "confidence": confidence
        }

# Global model instance
vit_model = None

def get_vit_model():
    """Get the global ViT model instance"""
    global vit_model
    if vit_model is None:
        vit_model = VitInference()
    return vit_model

def analyze_food_image(image: Image.Image):
    """Main function to analyze food image using ViT model"""
    model = get_vit_model()
    return model.predict(image)