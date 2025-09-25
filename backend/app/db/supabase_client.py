from supabase import create_client, Client
from dotenv import load_dotenv
import os
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_image_bytes(bucket: str, path: str) -> bytes:
    """Download an image from Supabase storage and return as bytes."""
    res = supabase.storage.from_(bucket).download(path)
    return res