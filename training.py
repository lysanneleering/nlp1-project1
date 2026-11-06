"""
Trains and evaluates classifiers using encoders fitted on the training fold
only.

Copied from nlp-illc/nlp1-student/part1/smlp.py:
    SparseDataset, scipy_to_torch_sparse, sparse_collate, train_one_epoch,
    predict_probs.

train_logistic_regression slightly adapts part1/loglin.py::fit.
train_neural_network_model adapts part1/smlp.py's training wrapper:
    supports both MLPs, seeds randomness, and uses fixed epochs.

predict combines course prediction code for logistic regression and MLPs.
score_predictions is self-written.
"""

import random
import numpy as np
import time
import scipy.sparse
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from models import create_logistic_regression, SparseMLP, SparseMLP2


class SparseDataset(Dataset):
    """Wraps a SciPy sparse feature matrix and label list for use with
    DataLoader.

    Inputs: X (sparse matrix of examples), y (integer labels).
    Output per item: one sparse row and its label tensor.
    """

    def __init__(self, X, y):
        self.X = X
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def scipy_to_torch_sparse(x):
    """Converts a SciPy sparse matrix to a coalesced PyTorch sparse COO tensor.

    Input: SciPy sparse matrix.
    Output: torch sparse tensor with the same shape and non-zero values.
    """
    x = x.tocoo()
    indices = torch.tensor([x.row, x.col], dtype=torch.long)
    values = torch.tensor(x.data, dtype=torch.float32)

    return torch.sparse_coo_tensor(indices, values, size=x.shape).coalesce()

def sparse_collate(batch):
    """Collates individual sparse rows into one PyTorch sparse batch.

    Input: list of (sparse row, label) pairs from SparseDataset.
    Process: stack SciPy rows, convert to a torch sparse tensor, stack labels.
    Output: (X_batch sparse tensor, y_batch long tensor).
    """
    X_rows, y_rows = zip(*batch)

    # Stack SciPy sparse rows, then convert the whole batch at once.
    X_batch = scipy.sparse.vstack(X_rows)
    X_batch = scipy_to_torch_sparse(X_batch)

    y_batch = torch.stack(y_rows)

    return X_batch, y_batch

def train_one_epoch(model, dataloader, optimizer, criterion, device="cpu"):
    """Run one training epoch: forward, loss, backward, optimizer step.

    Inputs: model, train dataloader, optimizer, loss function, device.
    Output: dict with average train loss and accuracy for the epoch.
    """
    if isinstance(device, str):
        device = torch.device(device)

    model.train()  # enable dropout / training behaviour

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for X_batch, y_batch in dataloader:
        # move the batch to the device
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()  # clear gradients from the previous step

        logprobs = model(X_batch) # forward pass
        loss = criterion(logprobs, y_batch) # compute the loss

        loss.backward()   # compute gradients
        optimizer.step()  # update parameters

        # accumulate weighted loss and correct predictions for epoch averages
        total_loss += loss.item() * y_batch.size(0)
        predictions = logprobs.argmax(dim=1) # get the predicted class
        total_correct += (predictions == y_batch).sum().item() # count the number of correct predictions
        total_examples += y_batch.size(0) # count the number of examples

    return {
        "loss": total_loss / total_examples, # average loss
        "accuracy": total_correct / total_examples, # average accuracy
    }

def predict_probs(texts, model, encoder, batch_size=10):
    """
    Returns a numpy array with shape [num_texts, num_classes] with the pmfs of
    the input texts as predicted by the model.

    texts: list of texts, each a string
    model: trained model
    encoder: encoder used in training
    batch_size: how many at a time
    """

    device = model.weight.device

    vectors = [encoder(x) for x in texts]
    labels = [0 for _ in texts]

    x_sparse = csr_matrix(vectors)

    dl = DataLoader(
        SparseDataset(x_sparse, labels),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sparse_collate,
    )

    model.eval()  # disable dropout / training behaviour

    all_probs = []
    with torch.no_grad():  # no gradients needed for evaluation

        for x_batch, y_batch in dl:
            # move the batch to the device
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logprobs = model(x_batch) # forward pass
            # accumulate metrics; also keep predictions for later use
            all_probs.extend(logprobs.exp().detach().cpu().numpy())

    return np.array(all_probs)

def train_logistic_regression(training_set, encoder, max_iter=1000, seed=13, C=1.0):
    """
    Constructs, trains and returns a softmax regressor.

    training_set: list of text-label pairs (a text is a string, a label is a 0-based integer)
    encoder: a fitted representation object, such as TokenIndicator
    """
    training_set = list(training_set)

    # Convert the training texts to vectors
    training_vectors = [encoder(text) for text, label in training_set]
    training_labels = [label for text, label in training_set]

    # To train the model more efficiently, we transform the training vectors to
    # a sparse matrix
    X_train_sparse = csr_matrix(training_vectors)

    # Train the logistic model
    logistic_model = create_logistic_regression(max_iter=max_iter, seed=seed, C=C)
    logistic_model.fit(X_train_sparse, training_labels)

    return logistic_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_neural_network_model(
    training_set, encoder, num_classes, model_option='mlp',
    hidden_dim=128, hidden_dim2=64, dropout=0.2, batch_size=64,
    learning_rate=1e-3, weight_decay=1e-4, num_epochs=20, seed=13,
    device='cpu', verbose=True,
):
    """
    Adapts the course wrapper: return (trained model, training history).

    The original selects a checkpoint using validation loss. Here we train for
    a fixed number of epochs and evaluate separately, so the held-out fold does
    not influence model selection.
    """
    set_seed(seed)
    if isinstance(device, str):
        device = torch.device(device)

    time_start = time.time()
    # Convert the training sets to vectors
    training_vectors = [encoder(text) for text, label in training_set]
    training_labels = [int(label) for text, label in training_set]

    if verbose:
        time_document_to_vector = time.time()
        print(f"Document transformation done! Time taken: {time_document_to_vector - time_start:.3f} seconds")

    # Keep SciPy sparse here; conversion to torch happens in sparse_collate
    X_train_sparse = csr_matrix(training_vectors)
    train_dataset = SparseDataset(X_train_sparse, training_labels)

    # create the data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=sparse_collate,
    )

    if verbose:
        time_dataloader = time.time()
        print(f"Dataloader done! Time taken: {time_dataloader - time_document_to_vector:.3f} seconds")

    # Project addition: choose either the course MLP or our two-layer extension.
    args = dict(input_dim=len(encoder), hidden_dim=hidden_dim,
                num_classes=num_classes, dropout=dropout)
    if model_option == 'mlp':
        model = SparseMLP(**args).to(device)
    else:
        model = SparseMLP2(**args, hidden_dim2=hidden_dim2).to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )

    history = []
    for epoch in range(num_epochs):
        # Use the original course function for every training epoch.
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        history.append({'epoch': epoch + 1, **train_metrics})
        if verbose:
            print(f"Epoch {epoch + 1:02d}/{num_epochs} | "
                  f"train loss: {train_metrics['loss']:.4f} | "
                  f"train accuracy: {train_metrics['accuracy']:.4f}", flush=True)

    if verbose:
        time_training = time.time()
        print(f"Training done! Time taken: {time_training - time_dataloader:.3f} seconds")

    # These are training-mode metrics, collected while parameters are changing.
    return model, history

def predict(texts, model, encoder, batch_size=64):
    """
    Predicts labels in input order for either model type.
    """
    texts = list(texts)
    if not texts:
        return np.empty(0, dtype=np.int64)

    if isinstance(model, nn.Module):
        # The course helper returns probabilities: select their argmax.
        probs = predict_probs(texts, model, encoder, batch_size=batch_size)
        return probs.argmax(axis=1).astype(np.int64)

    # This part follows loglin.py::predict.
    validation_vectors = [encoder(text) for text in texts]
    X_val_sparse = csr_matrix(validation_vectors)
    predictions = model.predict(X_val_sparse)

    return predictions.astype(np.int64)

def score_predictions(gold_labels, predictions, num_classes):
    """
    Calculates accuracy and macro-F1 over the full label set.
    """
    return {
        'accuracy': float(accuracy_score(gold_labels, predictions)),
        'macro_f1': float(f1_score(
            gold_labels, predictions, labels=list(range(num_classes)),
            average='macro', zero_division=0,
        )),
    }
