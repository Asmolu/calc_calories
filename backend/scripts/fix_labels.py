from transformers import ViTForImageClassification
from datasets import load_dataset

SRC_MODEL = "./vit-food101/model"
DST_MODEL = "./vit-food101/model-fixed"

# загрузка датасета
ds = load_dataset("food101", split="train")
labels = ds.features["label"].names

# загрузка модели
model = ViTForImageClassification.from_pretrained(SRC_MODEL)

# правка label mapping
model.config.id2label = {i: label for i, label in enumerate(labels)}
model.config.label2id = {label: i for i, label in enumerate(labels)}

# СОХРАНЯЕМ В ДРУГУЮ ПАПКУ
model.save_pretrained(DST_MODEL)

print("Saved fixed model to:", DST_MODEL)
print("Example label 25:", model.config.id2label[25])
