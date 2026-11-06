"""
Classification models.

Logistic regression uses scikit-learn just like the course.
SparseMLP copied from nlp-illc/nlp1-student/part1/smlp.py.
SparseMLP2 is a two-hidden-layer MLP, based on SparseMLP.
"""

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression


def create_logistic_regression(max_iter=1000, seed=13, C=1.0):
    """Creates a log-linear classifier using scikit-learn."""
    return LogisticRegression(
        solver="saga",
        max_iter=max_iter,
        random_state=seed,
        C=C,
    )


class SparseMLP(nn.Module):
    """One-hidden-layer MLP that accepts sparse input tensors.

    Inputs: feature vectors of size `input_dim` (e.g. bag-of-words).
    Process: sparse matmul into a dense hidden layer, ReLU + dropout, linear
    (num_classes) and log-softmax outputs
        log-softmax, rather than softmax, for numerical stability
    Output: logprobs over `num_classes`.
    """

    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.2):
        """
        input_dim: size of the input feature space (e.g. vocabulary size)
        hidden_dim: width of the hidden layer
        num_classes: number of output labels
        dropout: fraction of hidden units zeroed during training (regularisation)
        """
        super().__init__()

        # First-layer weights stay dense; inputs arriving here are sparse.
        self.weight = nn.Parameter(
            torch.empty(input_dim, hidden_dim)
        )
        self.bias = nn.Parameter(
            torch.zeros(hidden_dim)
        )

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Everything after the first layer is dense.
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
            nn.LogSoftmax(dim=-1) # for numerical stability, it is common to use log softmax, rather than softmax
        )

        nn.init.xavier_uniform_(self.weight) # why xavier?

    def forward(self, x):
        """Map a sparse batch [batch, input_dim] to class log-probabilities [batch, num_classes]."""
        # First matmul keeps x sparse; result is dense from here on.
        x = torch.sparse.mm(x, self.weight)
        x = x + self.bias

        x = self.activation(x)
        x = self.dropout(x)
        # logprobs
        return self.output(x)


class SparseMLP2(nn.Module):
    """Two-hidden-layer MLP that accepts sparse input tensors.

    Inputs: feature vectors of size `input_dim` (e.g. bag-of-words).
    Process: sparse matmul into the first hidden layer, ReLU + dropout,
    second hidden layer, ReLU + dropout, and log-softmax outputs.
    Output: logprobs over `num_classes`.
    """

    def __init__(self, input_dim, hidden_dim, num_classes, hidden_dim2=64, dropout=0.2):
        """
        input_dim: size of the input feature space (e.g. vocabulary size)
        hidden_dim: width of the first hidden layer
        hidden_dim2: width of the second hidden layer
        num_classes: number of output labels
        dropout: fraction of hidden units zeroed during training
        """
        super().__init__()

        # First-layer
        self.weight = nn.Parameter(
            torch.empty(input_dim, hidden_dim)
        )
        self.bias = nn.Parameter(
            torch.zeros(hidden_dim)
        )

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Second hidden layer
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim2)

        # Output layer
        self.output = nn.Sequential(
            nn.Linear(hidden_dim2, num_classes),
            nn.LogSoftmax(dim=-1)
        )

        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        """Map sparse input vectors to class log-probabilities."""

        # First hidden layer
        x = torch.sparse.mm(x, self.weight)
        x = x + self.bias
        x = self.activation(x)
        x = self.dropout(x)

        # Second hidden layer
        x = self.hidden2(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Output log-probabilities
        return self.output(x)
