# Summarise accuracy and macro-F1 scores
# Perform Bayesian correlated-t comparison with justified ROPE

from pathlib import Path
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
import analysis

results_dir = Path(__file__).resolve().parent / 'results'
rope = 0.005

files = sorted(results_dir.glob('cross_validation_*.csv'))

results_path = files[-1]

results = pd.read_csv(results_path)
results['system'] = results['representation'] + ' + ' + results['model']

# Summarise accuracy and macro-F1 scores
summary = results.groupby('system').agg(
    accuracy_mean=('accuracy', 'mean'),
    accuracy_std=('accuracy', 'std'),
    macro_f1_mean=('macro_f1', 'mean'),
    macro_f1_std=('macro_f1', 'std')
).sort_values('macro_f1_mean', ascending=False)

print('Performance summary\n')
print(summary.round(3))

summary.to_csv(results_dir / 'summary.csv')

# Perform Bayesian correlated-t comparison with justified ROPE
# Collect the 25 macro-F1 scores for each system, with the same repetition and fold order
scores = results.pivot(
    index=['repetition', 'fold'],
    columns='system',
    values='macro_f1'
).sort_index()

comparisons = []

best_system = summary.index[0]
second_best = summary.index[1]

# Compare all pairs of representation-model combinations
for system_a, system_b in combinations(scores.columns, 2):
    # Using the tutorials Bayesian correlated t model
    posterior = analysis.BenavoliCorrelatedT(
        scores[system_a].to_numpy(),
        scores[system_b].to_numpy(),
        runs=5,
        names=[system_a, system_b]
    )
    # Posterior probabilities for the 3 ROPE regions
    p_a, p_tie, p_b = posterior.probs(rope)

    # Plot the Bayesian posterior for the two best systems
    if {system_a, system_b} == {best_system, second_best}:
        print('\nBayesian comparison of the two best systems\n')
        print(f'A: {system_a}')
        print(f'B: {system_b}')
        print(f'P(A better): {p_a:.3f}')
        print(f'P(Equivalent): {p_tie:.3f}')
        print(f'P(B better): {p_b:.3f}\n')
        fig, ax = posterior.plot(rope)
        fig.savefig(
            results_dir / 'bayesian_best_vs_second.png',
        )
        plt.show()

    comparisons.append({
        'system_a': system_a,
        'system_b': system_b,
        'p_a_better': p_a,
        'p_equivalent': p_tie,
        'p_b_better': p_b
    })

comparisons = pd.DataFrame(comparisons)

comparisons.to_csv(
    results_dir / 'bayesian_comparisons.csv',
    index=False
)
