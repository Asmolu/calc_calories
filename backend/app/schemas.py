from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class MacroTotals(BaseModel):
    calories: float = Field(..., description="Total calories in kcal")
    proteins: float = Field(..., description="Total protein in grams")
    fats: float = Field(..., description="Total fat in grams")
    carbs: float = Field(..., description="Total carbohydrates in grams")


class RecognizedItem(BaseModel):
    name: str
    calories: float
    proteins: float
    fats: float
    carbs: float
    confidence: Optional[float] = Field(
        default=None, description="Optional confidence score returned by the model"
    )


class MealSummary(BaseModel):
    id: str
    created_at: datetime
    items: List[RecognizedItem]
    totals: MacroTotals


class PredictionResponse(BaseModel):
    items: List[RecognizedItem]
    totals: MacroTotals


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str