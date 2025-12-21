from __future__ import annotations

from typing import List, Dict
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE_DIR / "app/services/ai/models/vit_food101/model-fixed"
MODEL_PATH = str(MODEL_PATH)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ViTFoodClassifierPT:
    def __init__(self):
        self.processor = ViTImageProcessor.from_pretrained(
            MODEL_PATH,
            local_files_only=True
        )

        self.model = ViTForImageClassification.from_pretrained(
            MODEL_PATH,
            local_files_only=True
        )
        self.model.to(DEVICE)
        self.model.eval()

        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        top_k: int = 3,
    ) -> List[Dict]:
        inputs = self.processor(
            images=image,
            return_tensors="pt",
        ).to(DEVICE)

        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

        values, indices = torch.topk(probs, k=top_k)

        results = []
        for score, idx in zip(values[0], indices[0]):
            results.append(
                {
                    "label": self.id2label[idx.item()],
                    "confidence": float(score.item()),
                }
            )

        return results
