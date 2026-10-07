from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.router import api_router
from app.db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handling database table initialization.
    """
    await init_db()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=(
        "Backend API for the Multimodal Analysis of Communication and Language Patterns "
        "for Personalized Mental Health Risk Screening. "
        "This platform is an analytical pre-screening research system and does NOT provide clinical diagnoses."
    ),
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure Cross-Origin Resource Sharing (CORS)
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Mount API routes at /api/v1
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint returning service status and academic pointer.
    """
    return {
        "project": settings.PROJECT_NAME,
        "system_type": settings.SYSTEM_TYPE,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "health_check": f"{settings.API_V1_STR}/health",
        "disclaimer": settings.CLINICAL_DISCLAIMER
    }
