"""
Shared FastAPI dependencies
"""
from .db_models import SessionLocal
from .config import settings
from fastapi import HTTPException, Header
from typing import Optional

def get_db():
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def verify_admin_key(x_api_key: Optional[str] = Header(None)):
    """Verify admin API key for protected endpoints"""
    if not settings.ADMIN_API_KEY:
        return True  # If no key set, allow access
    
    if x_api_key != settings.ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True