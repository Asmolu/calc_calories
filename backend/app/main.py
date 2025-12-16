from fastapi import FastAPI

from app.api import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    configure_logging()
    settings = get_settings()

    app = FastAPI(title=settings.app_name, version=settings.version)
    app.include_router(api_router, prefix=settings.api_prefix)

    @app.get("/", tags=["health"])
    async def root():
        return {"message": f"{settings.app_name} backend is running"}

    return app


app = create_app()