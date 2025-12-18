import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

MODEL_PATH = "./vit-food101/model-fixed"
IMAGE_PATH = "food.png"  # любое фото еды

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
model = ViTForImageClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

image = Image.open(IMAGE_PATH).convert("RGB")
inputs = processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    top = torch.argmax(probs, dim=1).item()
    confidence = probs[0, top].item()

label = model.config.id2label[top]

print(f"Prediction: {label}, confidence={confidence:.3f}")
