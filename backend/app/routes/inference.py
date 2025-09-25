from fastapi import APIRouter, HTTPException
from app.models.schemas import AIImageAnalysisRequest, AIImageAnalysisResponse
import base64
import io
from PIL import Image
from app.db.supabase_client import supabase
from app.ml.vit_inference import analyze_food_image

router = APIRouter()

@router.post("/analyze-image")
async def analyze_food_image(request: AIImageAnalysisRequest):
    """
    Analyze food image stored in Supabase and return nutritional information
    """
    try:
        # Step 1: Fetch file from Supabase storage
        res = supabase.storage.from_(request.bucket).download(request.path)
        if res is None:
            raise HTTPException(status_code=404, detail="Image not found in Supabase bucket")

        # Step 2: Convert to PIL Image for AI model
        image = Image.open(io.BytesIO(res))

        # Step 3: Call your AI model
        ai_results = analyze_food_image(image)

        # Step 4: Return structured results to frontend
        return AIImageAnalysisResponse(
            food_name=ai_results["food_name"],
            calories=ai_results["calories"],
            protein=ai_results["protein"],
            carbs=ai_results["carbs"],
            fat=ai_results["fat"],
            confidence=ai_results["confidence"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")

# Add endpoint for direct base64 image analysis
@router.post("/analyze-base64")
async def analyze_base64_image(image_base64: str):
    """
    Analyze food image from base64 string directly
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))

        # Call AI model
        ai_results = analyze_food_image(image)

        return ai_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")
