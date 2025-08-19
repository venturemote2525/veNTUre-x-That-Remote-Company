
# dummy data for now
from fastapi import FastAPI
from .db.supabase_client import supabase  

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello from FastAPI + Supabase!"}

@app.post("/add_dummy")
def add_dummy():
    data = {"name": "chickenrice", "calories": 250}
    supabase.table("foods").insert(data).execute()
    return {"status": "inserted", "data": data}

@app.get("/get_dummy")
def get_dummy():
    response = supabase.table("foods").select("*").execute()
    return response.data