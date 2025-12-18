import os
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)

MODEL_NAME = "google/vit-base-patch16-224"
OUT_DIR = "./vit-food101"

# ----------------------
# 1) Dataset
# ----------------------
dataset = load_dataset("food101")  # train + validation (у HF Food101 test обычно отсутствует)

# ----------------------
# 2) Processor
# ----------------------
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

# ----------------------
# 3) Robust transform (batched-safe + RGB-safe)
# ----------------------
def _to_rgb(img):
    # HF Food101 обычно PIL.Image.Image, но иногда может быть массив/нестандарт
    # ViT expects 3-channel images.
    try:
        # PIL Image
        if hasattr(img, "mode"):
            if img.mode != "RGB":
                return img.convert("RGB")
            return img
    except Exception:
        pass

    # numpy array fallback
    arr = np.asarray(img)
    if arr.ndim == 2:
        # grayscale -> 3-channel
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr

def transform(batch):
    # batch может быть либо одним примером, либо батчом (dict of lists)
    images = batch["image"]
    labels = batch["label"]

    if not isinstance(images, (list, tuple)):
        images = [images]
        labels = [labels]

    images = [_to_rgb(im) for im in images]

    inputs = processor(images=images, return_tensors="pt")

    # ВАЖНО: вернуть "labels" и "pixel_values"
    # labels должен быть тензором длины B
    return {
        "pixel_values": inputs["pixel_values"],          # (B, 3, 224, 224)
        "labels": torch.tensor(labels, dtype=torch.long) # (B,)
    }

# set_transform -> трансформ применяется при чтении элементов
dataset.set_transform(transform)

# ----------------------
# 4) Model
# ----------------------
model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=101,
    ignore_mismatched_sizes=True,
)

# ----------------------
# 5) Metrics
# ----------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": float((preds == labels).mean())}

# ----------------------
# 6) Training args
# ----------------------
cuda_ok = torch.cuda.is_available()
print("CUDA available:", cuda_ok)
if cuda_ok:
    print("CUDA device:", torch.cuda.get_device_name(0))

# fp16 обычно ок; если захочешь bf16 — можно добавить проверку.
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    eval_strategy="steps",
    save_strategy="steps",

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,

    num_train_epochs=2,
    learning_rate=2e-5,

    fp16=cuda_ok,
    # bf16=False,  # при желании можно включить, если захочешь

    logging_steps=50,
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,

    remove_unused_columns=False,
    report_to="none",

    # Windows-friendly
    dataloader_num_workers=0,
    dataloader_pin_memory=cuda_ok,
)

# ----------------------
# 7) Trainer
# ----------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=DefaultDataCollator(),
    compute_metrics=compute_metrics,
)

# ----------------------
# 8) Train
# ----------------------
trainer.train()

# ----------------------
# 9) Save
# ----------------------
save_path = os.path.join(OUT_DIR, "model")
model.save_pretrained(save_path)
processor.save_pretrained(save_path)
print("Saved to:", save_path)
