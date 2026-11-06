# cross_validation_split helper adapted from nlp-illc/nlp1-student

from datasets import load_dataset
from sklearn.model_selection import StratifiedKFold
import itertools
import json

def cross_validation_split(pairs, k, seed=0):
    """
    Shuffles labelled texts and splits them into k stratified folds.

    pairs: list of tuples (text, label)
    k: number of folds
    seed: for reproducible shuffling
    """

    pairs = list(pairs)
    ids = list(range(len(pairs)))
    labels = [label for _, label in pairs]

    # Shuffle and split while preserving approximate class proportions
    splitter = StratifiedKFold(
        n_splits=k,
        shuffle=True,
        random_state=seed,
    )
    ids_parts = [
        validation_ids
        for _, validation_ids in splitter.split(ids, labels)
    ]

    # Create a dictionary to store the training and validation sets for each fold
    cross_validation_sets = {}
    for i in range(k):
        validation_set = [pairs[j] for j in ids_parts[i]]
        training_ids = list(
            itertools.chain.from_iterable(
                part for k, part in enumerate(ids_parts) if k != i
            )
        )
        training_set = [pairs[j] for j in training_ids]
        cross_validation_sets[i] = {
            "training": training_set,
            "validation": validation_set,
            "training_ids": training_ids,
            "validation_ids": ids_parts[i],
        }

    return cross_validation_sets

data = load_dataset("cardiffnlp/tweet_eval", "emotion")

# Define the random seeds and number of folds
my_seeds = [0, 1, 2, 3, 4]
num_folds = 5

# Store the cross-validation folds for each seed
my_cross_validation_sets = []

# Generate cross-validation folds for each seed
for seed in my_seeds:
    # Use the tutorial helper to create N folds
    splits = cross_validation_split(
        # Pair tweet with its label
        pairs=list(zip(data["train"]["text"], data["train"]["label"])),
        k=num_folds,
        seed=seed,
    )

    # Store the training and validation indices
    folds = []

    for fold in range(num_folds):
        train_idx = splits[fold]["training_ids"]
        eval_idx = splits[fold]["validation_ids"]

        # Convert indices to integers so they can be saved as json
        folds.append((
            [int(i) for i in train_idx],
            [int(i) for i in eval_idx],
        ))

    my_cross_validation_sets.append(folds)

# Save all folds so everyone uses identical splits
with open("cross_validation_sets.json", "w") as file:
    json.dump(my_cross_validation_sets, file)

print("Cross-validation folds saved")