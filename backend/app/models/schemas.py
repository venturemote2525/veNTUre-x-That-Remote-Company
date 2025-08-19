# for response validaiton
from pydantic import BaseModel, EmailStr
from typing import Optional

# --- Auth ---
class SignupRequest(BaseModel):
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

# --- Food ---
class FoodItem(BaseModel):
    name: str
    calories: int
