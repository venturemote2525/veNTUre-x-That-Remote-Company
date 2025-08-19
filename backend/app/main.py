
# dummy data for now
from fastapi import FastAPI
from .db.supabase_client import supabase  
from .routes import food, auth

app = FastAPI()

# Include routers
app.include_router(food.router, prefix="/food", tags=["food"])
app.include_router(auth.router, prefix="/auth", tags=["auth"])

@app.get("/")
def root():
    return {"message": "Hello from FastAPI + Supabase!"}

