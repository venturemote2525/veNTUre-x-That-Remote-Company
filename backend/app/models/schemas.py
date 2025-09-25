# for response validaiton
from pydantic import BaseModel, EmailStr
from typing import Optional

# auth
class SignupRequest(BaseModel):
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

# food
class FoodItem(BaseModel):
    name: str
    calories: int

class AIImageAnalysisRequest(BaseModel):
    bucket: str
    path: str
    user_id: Optional[str] = None
    meal_type: Optional[str] = None

#  #what frontend sends to backend
# class AIImageAnalysisRequest(BaseModel):
#     image_base64: str  # Base64 encoded image from frontend
#     user_id: Optional[str] = None # Optional user ID for personalized analysis
#     meal_type: Optional[str] = None # Optional meal type (breakfast, lunch, dinner)

# response from backend to frontend
class MealItem(BaseModel):
    name: str
    image_url: str
    meal: Optional[str]  # USER-DEFINED type in Postgres, so keep it flexible as str
    calories: int
    protein: int
    carbs: int
    fat: float