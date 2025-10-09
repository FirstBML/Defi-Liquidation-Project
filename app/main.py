"""
Enhanced FastAPI Application for DeFi Liquidation Risk System
Ready for Railway, ngrok, and Vercel Integration
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

# Load environment variables FIRST
load_dotenv()

# ------------------------------------------------------
# 🔹 Application Lifespan (startup/shutdown events)
# ------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[{datetime.now(timezone.utc)}] Starting up FastAPI app...")
    yield
    print(f"[{datetime.now(timezone.utc)}] Shutting down FastAPI app...")


# ------------------------------------------------------
# 🔹 Initialize FastAPI App
# ------------------------------------------------------
app = FastAPI(
    title="DeFi Risk Early-Warning System",
    description="Real-time liquidation risk monitoring for DeFi protocols",
    version="2.0.0",
    lifespan=lifespan,
)


# ------------------------------------------------------
# 🔹 CORS Configuration (for Vercel + ngrok + localhost)
# ------------------------------------------------------
ALLOWED_ORIGINS = [
    "https://perspectively-slaty-sheilah.ngrok-free.dev",  # your ngrok public URL
    "https://your-vercel-dashboard.vercel.app",             # replace with your real Vercel dashboard URL
    "http://localhost:3000",                                # local dev frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------
# 🔹 Routers (if you have separate API route files)
# ------------------------------------------------------
# Example (uncomment when routes exist)
# from app.routes import alerts, aave, positions
# app.include_router(alerts.router, prefix="/alerts")
# app.include_router(aave.router, prefix="/aave")
# app.include_router(positions.router, prefix="/positions")


# ------------------------------------------------------
# 🔹 Root & Health Endpoints
# ------------------------------------------------------
@app.get("/")
def root():
    return {"message": "DeFi Risk Early-Warning System API is running 🚀"}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ------------------------------------------------------
# 🔹 Run (for local testing)
# ------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
