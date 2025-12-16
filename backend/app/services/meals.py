from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

from app.schemas import MealSummary, RecognizedItem
from app.services.utils import calculate_totals


def _demo_items() -> List[RecognizedItem]:
    return [
        RecognizedItem(name="Grilled chicken", calories=230, proteins=35, fats=5, carbs=0),
        RecognizedItem(name="Quinoa", calories=120, proteins=4, fats=2, carbs=21),
        RecognizedItem(name="Avocado", calories=90, proteins=1, fats=8, carbs=5),
    ]


def get_recent_meals() -> List[MealSummary]:
    """Return a small set of demo meals to populate the UI."""

    first_items = _demo_items()
    second_items = [
        RecognizedItem(name="Pancakes", calories=310, proteins=8, fats=10, carbs=48),
        RecognizedItem(name="Greek yogurt", calories=100, proteins=17, fats=0, carbs=6),
    ]

    now = datetime.utcnow()
    meals = [
        MealSummary(
            id="demo-1",
            created_at=now - timedelta(hours=1),
            items=first_items,
            totals=calculate_totals(first_items),
        ),
        MealSummary(
            id="demo-2",
            created_at=now - timedelta(days=1, hours=2),
            items=second_items,
            totals=calculate_totals(second_items),
        ),
    ]
    return meals