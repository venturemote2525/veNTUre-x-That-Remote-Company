from fastapi import APIRouter, HTTPException
from app.models.schemas import InferenceResponse, InferenceRequest
import base64
import io
from PIL import Image
from app.db.supabase_client import supabase
from app.ml.vit_inference import analyze_food_image
from supabase import create_client, Client
import os

service_supabase: Client = create_client(
    os.getenv("SUPABASE_URL"), 
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # This bypasses RLS
)

router = APIRouter()

@router.post("/analyze-image", response_model=InferenceResponse)
async def analyze_food_image_endpoint(request: InferenceRequest):
    print(f"=== DEBUG INFO ===")
    print(f"Request: bucket={request.bucket}, path={request.path}")
    print(f"User ID: {request.user_id}")
    print(f"Meal type: {request.meal_type}")
    files = service_supabase.storage.from_("meal_images").list("bf760144-f5ed-42e4-abdc-9a10a994fdd5")
    print("Files in user folder:", [f["name"] for f in files])

    try:
        # Test basic Supabase connection
        try:
            # Try to list buckets first
            buckets = supabase.storage.list_buckets()
            print(f"Available buckets: {buckets}")
        except Exception as bucket_error:
            print(f"Error listing buckets: {bucket_error}")
            print(f"Bucket error type: {type(bucket_error)}")
        
        # Try to list files in the specific bucket
        try:
            files = supabase.storage.from_(request.bucket).list()
            print(f"Files in '{request.bucket}': {files}")
        except Exception as list_error:
            print(f"Error listing files in bucket '{request.bucket}': {list_error}")
            print(f"List error type: {type(list_error)}")
        
        # Try the actual download
        print(f"Attempting download...")
        res = service_supabase.storage.from_(request.bucket).download(request.path)
        print(f"Download result type: {type(res)}")
        print(f"Download result: {res[:100] if isinstance(res, bytes) else res}")
        
        if res is None:
            raise HTTPException(status_code=404, detail="Image not found in Supabase")

        # Convert bytes → PIL image
        image = Image.open(io.BytesIO(res))
        print(f"Successfully opened image: {image.size}")

        # Call AI model
        ai_results = analyze_food_image(image)
        
        return InferenceResponse(**ai_results)

    except Exception as e:
        print(f"=== FULL ERROR DETAILS ===")
        print(f"Error: {e}")
        print(f"Error type: {type(e)}")
        print(f"Error args: {e.args}")
        
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        
        raise HTTPException(status_code=500, detail=f"AI inference failed: {str(e)}")

def call_food_ai_model():
    """
    Replace this with your actual ONNX / FastAI inference
    """
    # Temporary dummy data
    return {
        "food_name": "Chicken Rice",
        "calories": 650,
        "protein": 25,
        "carbs": 45,
        "fat": 15,
        "confidence": 0.85,
    }
