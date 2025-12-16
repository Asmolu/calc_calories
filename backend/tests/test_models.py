from datetime import datetime

from app.schemas import HealthResponse, MacroTotals, MealSummary, PredictionResponse, RecognizedItem
from app.services.utils import calculate_totals


def test_calculate_totals_rounds_values():
    items = [
        RecognizedItem(name="Item A", calories=100.456, proteins=10.123, fats=5.987, carbs=12.333),
        RecognizedItem(name="Item B", calories=50.4, proteins=2.2, fats=1.1, carbs=3.7),
    ]

    totals = calculate_totals(items)

    assert totals.calories == 150.86
    assert totals.proteins == 12.32
    assert totals.fats == 7.09
    assert totals.carbs == 16.03


def test_prediction_and_meal_summary_models():
    items = [RecognizedItem(name="Salad", calories=120, proteins=5, fats=7, carbs=10, confidence=0.8)]
    totals = MacroTotals(calories=120, proteins=5, fats=7, carbs=10)
    response = PredictionResponse(items=items, totals=totals)

    assert response.items[0].confidence == 0.8
    assert response.totals.calories == 120

    meal = MealSummary(id="meal-1", created_at=datetime.utcnow(), items=items, totals=totals)
    assert meal.items == items
    assert meal.totals == totals


def test_health_response_structure():
    health = HealthResponse(status="ok", version="0.1.0", environment="testing")

    assert health.status == "ok"
    assert health.version == "0.1.0"
    assert health.environment == "testing"