🍽️ CalAI — Food Recognition Backend

CalAI is an AI-powered backend service for recognizing food dishes from images and estimating their nutritional values.

The project focuses on dish-level recognition using deep learning models rather than low-level ingredient detection.

✨ Features

📷 Image-based food dish recognition

🧠 Vision Transformer (ViT) fine-tuned on Food-101

⚡ GPU-accelerated training and inference (PyTorch + CUDA)

🌐 REST API built with FastAPI

📊 Nutrition estimation (calories, proteins, fats, carbs)

🧪 Fully covered with automated tests

🏗️ Architecture
Image
 └─▶ ViT (Food-101)
      └─▶ Dish classification (e.g. club_sandwich)
           └─▶ Nutrition lookup


The system intentionally avoids ingredient-level detection for better real-world accuracy.

🧠 Model

Base model: google/vit-base-patch16-224

Fine-tuned on: Food-101 dataset (101 dish classes)

Framework: PyTorch + HuggingFace Transformers

Inference: GPU / CPU supported

Model weights are not stored in this repository.

🚀 API Overview
Health Check
GET /api/health

Food Prediction
POST /api/ai/predict
Content-Type: multipart/form-data


Response example:

{
  "items": [
    {
      "name": "club_sandwich",
      "confidence": 0.986
    }
  ],
  "totals": {
    "calories": 420,
    "proteins": 28,
    "fats": 22,
    "carbs": 35
  }
}

🧪 Tests
pytest


Includes:

API tests

Model inference tests

Nutrition calculation tests

🛠️ Setup (Development)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt


Run backend:

uvicorn app.main:app --reload

⚠️ Notes

Model weights and datasets are intentionally excluded from Git

Training scripts are provided for reproducibility

GPU support requires a compatible CUDA-enabled PyTorch build

📄 License

Apache 2.0

🧠 Status

This project is under active development and serves as a foundation for AI-powered nutrition and food analysis systems.

