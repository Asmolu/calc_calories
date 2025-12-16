from fastapi import APIRouter, Depends, File, UploadFile

from app.core.config import get_settings
from app.schemas import HealthResponse, MealSummary, PredictionResponse, RecognizedItem
from app.services.meals import get_recent_meals
from app.services.utils import calculate_totals

api_router = APIRouter()


@api_router.get("/health", response_model=HealthResponse, tags=["health"])
async def health(settings=Depends(get_settings)) -> HealthResponse:  # type: ignore[override]
    """Return basic service information for uptime checks."""

    return HealthResponse(
        status="ok",
        version=settings.version,
        environment=settings.environment,
    )


@api_router.get("/meals/recent", response_model=list[MealSummary], tags=["meals"])
async def recent_meals() -> list[MealSummary]:
    """Serve a small list of demo meals for the UI."""

    return get_recent_meals()


@api_router.post("/ai/predict", response_model=PredictionResponse, tags=["ai"])
async def predict(file: UploadFile | None = File(default=None)) -> PredictionResponse:
    """Mock AI prediction endpoint.

    The endpoint accepts an optional image upload and returns deterministic
    nutritional information to allow the frontend to be exercised before the
    actual ONNX-backed inference is wired in.
    """

    placeholder_items = [
        RecognizedItem(name="Salmon", calories=180, proteins=20, fats=10, carbs=0, confidence=0.92),
        RecognizedItem(name="Roasted vegetables", calories=140, proteins=3, fats=7, carbs=18, confidence=0.74),
    ]

    totals = calculate_totals(placeholder_items)
    return PredictionResponse(items=placeholder_items, totals=totals)