# app/routes/auth.py
from fastapi import APIRouter
from app.models.schemas import SignupRequest, LoginRequest
from app.db.supabase_client import supabase

router = APIRouter()

@router.post("/signup")
def signup(payload: SignupRequest):
    try:
        user = supabase.auth.sign_up(
            {"email": payload.email, "password": payload.password}
        )
        return {"message": "Signup success", "data": user}
    except Exception as e:
        return {"error": str(e)}

@router.post("/login")
def login(payload: LoginRequest):
    try:
        user = supabase.auth.sign_in_with_password(
            {"email": payload.email, "password": payload.password}
        )
        return {"message": "Login success", "data": user}
    except Exception as e:
        return {"error": str(e)}
