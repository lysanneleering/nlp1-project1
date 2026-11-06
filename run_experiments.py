"""
Compares three representations and three classifiers over five repetitions of
five folds.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from datasets import load_dataset
from tokenization import train_tokenizer
from representations import TokenIndicator, BagOfWords, TFIDF
from training import (
    train_logistic_regression, train_neural_network_model,
    predict, score_predictions,
)

vocab_size = 1000
min_frequency = 2
model_seed = 13
split_seeds = [0, 1, 2, 3, 4]
num_epochs = 20
mlp_settings = dict(
    hidden_dim=128, hidden_dim2=64, dropout=0.2, batch_size=64,
    learning_rate=1e-3, weight_decay=1e-4, device='cpu',
)
lr_settings = dict(max_iter=1000, C=1.0)

# Load the emotion training split and count its label classes
data = load_dataset('cardiffnlp/tweet_eval', 'emotion', split='train')
num_classes = len(data.features['label'].names)

# Load the saved row indices that divide the data into cross-validation folds
project_dir = Path(__file__).resolve().parent
with (project_dir / 'cross_validation_sets.json').open(encoding='utf-8') as file:
    folds = json.load(file)

# Create a results folder if it does not already exist
results_dir = project_dir / 'results'
results_dir.mkdir(exist_ok=True)

# Include the current date and time in the filename to keep runs separate
stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
results_path = results_dir / f'cross_validation_{stamp}.csv'

# Define the CSV columns for experiment settings and evaluation scores
fields = [
    'dataset', 'task', 'repetition', 'fold', 'split_seed', 'model_seed',
    'representation', 'model', 'train_size', 'validation_size',
    'requested_vocab_size', 'actual_vocab_size', 'min_frequency',
    'num_epochs', 'settings', 'accuracy', 'macro_f1',
]

results = []
total_runs = 5 * 5 * 3 * 3
print(f'Writing results to: {results_path}')

with results_path.open('x', newline='', encoding='utf-8') as file:
    # Write dictionaries as CSV rows, using 'fields' as the column names
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    file.flush()

    # Loop through every repetition and its five saved folds
    for repetition_index, repetition in enumerate(folds):
        for fold_index, (training_ids, validation_ids) in enumerate(repetition):
            print(f'\n=== Repetition {repetition_index + 1}/5, '
                  f'fold {fold_index + 1}/5 ===', flush=True)

            # Select this fold's training and validation examples by row index
            training_fold = data.select(training_ids)
            validation_fold = data.select(validation_ids)

            # Prepare training (text, label) pairs and separate validation lists
            training_texts = list(training_fold['text'])
            training_set = list(zip(training_texts, training_fold['label']))
            validation_texts = list(validation_fold['text'])
            validation_labels = list(validation_fold['label'])
            print(f'Training examples: {len(training_set)}')
            print(f'Validation examples: {len(validation_texts)}')

            # Learn a new tokenizer from this fold's training texts only
            tokenizer = train_tokenizer(
                training_texts, vocab_size=vocab_size, min_frequency=min_frequency,
            )

            # Create the three representations and fit IDF on training texts only
            encoders = {
                'indicator': TokenIndicator(tokenizer),
                'bow': BagOfWords(tokenizer),
                'tfidf': TFIDF(tokenizer, training_texts),
            }

            # Train and evaluate each model with each representation
            for representation, encoder in encoders.items():
                for model_name in ['logistic_regression', 'mlp', 'mlp2']:
                    print(f'\nRun {len(results) + 1}/{total_runs}: '
                          f'repetition {repetition_index + 1}, fold {fold_index + 1} | '
                          f'{representation} + {model_name}', flush=True)

                    if model_name == 'logistic_regression':
                        model = train_logistic_regression(
                            training_set, encoder, seed=model_seed, **lr_settings,
                        )
                        settings = lr_settings
                    else:
                        model, history = train_neural_network_model(
                            training_set, encoder, num_classes,
                            model_option=model_name, num_epochs=num_epochs,
                            seed=model_seed, verbose=False, **mlp_settings,
                        )
                        settings = mlp_settings

                    # Predict held-out labels and calculate accuracy and macro-F1
                    predictions = predict(validation_texts, model, encoder)
                    scores = score_predictions(validation_labels, predictions, num_classes)

                    row = dict(
                        dataset='cardiffnlp/tweet_eval', task='emotion',
                        repetition=repetition_index + 1, fold=fold_index + 1,
                        split_seed=split_seeds[repetition_index], model_seed=model_seed,
                        representation=representation, model=model_name,
                        train_size=len(training_set), validation_size=len(validation_texts),
                        requested_vocab_size=vocab_size, actual_vocab_size=len(encoder),
                        min_frequency=min_frequency,
                        num_epochs='' if model_name == 'logistic_regression' else num_epochs,
                        settings=json.dumps(settings, sort_keys=True), **scores,
                    )

                    # Write the completed result immediately and keep it for the summary
                    writer.writerow(row)
                    file.flush()
                    results.append(row)

                    # Display this run's held-out scores
                    print(f"Accuracy: {scores['accuracy']:.3f} | "
                          f"Macro-F1: {scores['macro_f1']:.3f}")

# Summarize the 25 runs for each representation–model combination
print('\nCross-validation summary (25 scores per combination):')
for representation in ['indicator', 'bow', 'tfidf']:
    for model_name in ['logistic_regression', 'mlp', 'mlp2']:
        # Select the results belonging to this combination
        rows = [r for r in results if r['representation'] == representation
                and r['model'] == model_name]
        accuracies = [r['accuracy'] for r in rows]
        f1_scores = [r['macro_f1'] for r in rows]

        print(f'{representation:10} {model_name:20} '
              f'accuracy={mean(accuracies):.3f} +/- {stdev(accuracies):.3f}  '
              f'macro-F1={mean(f1_scores):.3f} +/- {stdev(f1_scores):.3f}')

print(f'\nSaved {len(results)} results to: {results_path}')