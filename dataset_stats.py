"""
Self-written code that finds dataset split sizes, training class counts,
and sample texts with their labels.
"""

from collections import Counter
from datasets import load_dataset

dataset = load_dataset("cardiffnlp/tweet_eval", "emotion")

# Show the sizes of the existing splits.
print("\nExisting Dataset Splits:\n")
for name, split in dataset.items():
    print(f"{name} - {len(split)} examples")

# Read the labels from the training dataset.
train = dataset["train"]
labels = train.features["label"].names

# Count how many examples are in each class.
print("\nTraining Examples Per Class:\n")
counts = Counter(train["label"])
for i, label in enumerate(labels):
    print(f"{label} - {counts[i]} examples")

# Print 5 training examples.
print("\nFive Training Examples:")
for example in train.shuffle(seed=13).select(range(5)):
    print(f"\nText: {example['text']}")
    print(f"Label: {labels[example['label']]}")
