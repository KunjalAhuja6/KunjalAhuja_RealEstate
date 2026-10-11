from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.utils.config import settings
from src.utils.logger import get_logger
from src.api.routes import router

logger = get_logger("Main")

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Microservice for AI-Powered Real Estate Property Valuation, Hybrid Retrieval, Multi-Agent Collaboration, and Quality Evaluation.",
)

# Enable CORS for frontend and API consumers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(router)

@app.get("/", tags=["Health"])
def root():
    return {
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "operational",
        "documentation_url": "/docs",
        "endpoints_prefix": "/api/v1",
    }

@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
    }

if __name__ == "__main__":
    logger.info(f"Starting API server on {settings.API_HOST}:{settings.API_PORT}...")
    uvicorn.run("src.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=False)
