
# dummy data for now
from fastapi import FastAPI
from .db.supabase_client import supabase  
from .routes import food

app = FastAPI()

# Include routers
app.include_router(food.router, prefix="/food", tags=["food"])

@app.get("/")
def root():
    return {"message": "Hello from FastAPI + Supabase!"}

