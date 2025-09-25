
# dummy data for now
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .db.supabase_client import supabase
from .routes import food, auth

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["exp://192.168.1.237:8081"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(food.router, prefix="/food", tags=["food"])
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(food.router, prefix="/inference", tags=["Inference"])

@app.get("/")
def root():
    return {"message": "Hello from FastAPI + Supabase!"}

