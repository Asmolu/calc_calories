from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.schemas import HealthResponse, MealSummary, PredictionResponse, RecognizedItem
from app.services.meals import get_recent_meals
from app.services.utils import calculate_totals
from app.services.ai import inference

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
    """Run AI inference on the uploaded food photo.

    The endpoint performs ONNXRuntime inference on the provided image bytes,
    converts detections to the public response schema, and computes aggregate
    nutrition totals. Nutrition values are zero-initialized until a nutrition
    database is integrated; the primary goal is to validate the AI pipeline end
    to end.
    """

    if file is None:
        raise HTTPException(status_code=400, detail="Image file is required")

    image_bytes = await file.read()
    detections = inference.predict(image_bytes=image_bytes)

    items = [
        RecognizedItem(
            name=det["class_name"],
            calories=0.0,
            proteins=0.0,
            fats=0.0,
            carbs=0.0,
            confidence=float(det.get("confidence", 0.0)),
            bounding_box=det.get("bounding_box"),
        )
        for det in detections
    ]

    totals = calculate_totals(items)
    return PredictionResponse(items=items, totals=totals)