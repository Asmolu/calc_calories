from transformers import ViTImageProcessor

SRC = "google/vit-base-patch16-224"
DST = "./vit-food101/model-fixed"

processor = ViTImageProcessor.from_pretrained(SRC)
processor.save_pretrained(DST)

print("Processor saved to:", DST)
