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
