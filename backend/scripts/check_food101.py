from datasets import load_dataset

dataset = load_dataset("food101")

print(dataset)
print("Train size:", len(dataset["train"]))
print("Validation size:", len(dataset["validation"]))
print("Example:", dataset["train"][0])
