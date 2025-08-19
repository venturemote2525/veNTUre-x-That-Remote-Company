# for response validaiton
from pydantic import BaseModel, EmailStr
from typing import Optional

class FoodItem(BaseModel):
    name: str
    calories: int