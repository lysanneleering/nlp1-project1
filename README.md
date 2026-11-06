# NLP1 Project 1

Emotion classification using the **emotion** subset of [TweetEval](https://huggingface.co/datasets/cardiffnlp/tweet_eval): anger, joy, optimism and sadness. We compare three text representations (token indicators, bag-of-words and TF-IDF) with three classifiers (logistic regression, a one-hidden-layer MLP and a two-hidden-layer MLP).

## Experiments

We use five repetitions of stratified five-fold cross-validation: **225 runs**, with 25 scores per representation–model combination. All systems use the same saved folds from the original training split. The separate validation and test splits are not used.

## Running the code

Run these commands from the project folder:

```bash
python3 -m pip install -r requirements.txt
python3 run_experiments.py
python3 analyze_results.py
```

Saved folds and results are included, so you can run `analyze_results.py` without retraining. It reads the newest `cross_validation_*.csv` in `results/`.

Optional commands:

```bash
python3 dataset_stats.py  # Display split sizes, label counts and examples
python3 prepare_folds.py  # Regenerate the saved cross-validation folds
```

## Results

`results/` contains individual run scores, a summary of mean accuracy and macro-F1 with standard deviations, Bayesian pairwise comparisons, and a posterior plot for the two highest-ranked systems.

The Bayesian correlated-t comparisons use five repetitions and a ROPE of ±0.005 macro-F1 (half a percentage point).

## Sources

Parts of the code are copied or adapted from [nlp-illc/nlp1-student](https://github.com/nlp-illc/nlp1-student), as indicated in the Python files.