from fastapi import APIRouter
from app.api.v1 import health, chat

api_router = APIRouter()

# Register Version 1 routes
api_router.include_router(health.router, prefix="", tags=["System & Health"])
api_router.include_router(chat.router, prefix="", tags=["Communication & WebSockets"])
