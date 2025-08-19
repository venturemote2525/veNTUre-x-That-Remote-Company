
from fastapi import APIRouter
from app.db.supabase_client import supabase
from app.models.schemas import FoodItem 

router = APIRouter()

@router.post("/add")
def add_food(item: FoodItem):
    data = {"name": item.name, "calories": item.calories}
    supabase.table("foods").insert(data).execute()
    return {"status": "inserted", "data": data}

@router.get("/all")
def get_foods():
    response = supabase.table("foods").select("*").execute()
    return {"data": response.data}
