from __future__ import annotations

from typing import Iterable

from app.schemas import MacroTotals, RecognizedItem


def calculate_totals(items: Iterable[RecognizedItem]) -> MacroTotals:
    """Aggregate nutritional totals for a list of recognized items."""

    calories = sum(item.calories for item in items)
    proteins = sum(item.proteins for item in items)
    fats = sum(item.fats for item in items)
    carbs = sum(item.carbs for item in items)

    return MacroTotals(
        calories=round(calories, 2),
        proteins=round(proteins, 2),
        fats=round(fats, 2),
        carbs=round(carbs, 2),
    )