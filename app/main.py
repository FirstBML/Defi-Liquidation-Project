# main.py

# app/main.py

from fastapi import FastAPI
from .api import router as api_router  # import router from your Liquidation/api.py

app = FastAPI(title="Liquidation Risk API")

# Register API routes
app.include_router(api_router)

@app.get("/")
def root():
    return {"message": "Liquidation Risk API is running"}


"""
import os
import uvicorn
from fastapi import FastAPI
from app.api import router as api_router
from app.scheduler import start_scheduler, stop_scheduler

app = FastAPI(title="Aave Risk Early-Warning System")

# include API routes under /api
app.include_router(api_router, prefix="/api")

@app.on_event("startup")
async def on_startup():
    # start scheduler background job
    start_scheduler()
    print("🚀 Scheduler started")

@app.on_event("shutdown")
async def on_shutdown():
    stop_scheduler()
    print("🛑 Scheduler stopped")

@app.get("/")
def root():
    return {"status": "ok", "service": "aave-risk"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
"""

