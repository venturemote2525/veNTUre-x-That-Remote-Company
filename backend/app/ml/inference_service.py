# from app.ml.inference import ONNXClassifier
# from app.db.supabase_client import supabase

# # Load model once at startup
# classifier = ONNXClassifier("app/ai_models/vit_small.onnx")

# def fetch_image_bytes(bucket: str, path: str) -> bytes:
#     """Download image from Supabase Storage"""
#     res = supabase.storage.from_(bucket).download(path)
#     return res  # already bytes

# def classify_image(bucket: str, path: str):
#     """Fetch image from Supabase and run inference"""
#     image_bytes = fetch_image_bytes(bucket, path)
#     result = classifier.predict(image_bytes)
#     return result
