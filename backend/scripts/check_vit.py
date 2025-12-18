from transformers import ViTImageProcessor, ViTForImageClassification
import torch

model_name = "google/vit-base-patch16-224"

processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

print("Model loaded")
print("Number of classes:", model.config.num_labels)
