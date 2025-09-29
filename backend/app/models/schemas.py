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

class InferenceRequest(BaseModel):
    bucket: str
    path: str
    user_id: Optional[str] = None
    meal_type: Optional[str] = None

class InferenceResponse(BaseModel):
    food_name: str
    calories: float
    protein: float
    carbs: float
    fat: float
    confidence: float