"""
Text representations for classification.

TokenIndicator slightly adapted from nlp-illc/nlp1-student.
BagOfWords and TFIDF are self-written.
"""

import numpy as np


class TokenIndicator:
    """Represents text by token presence."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Dimensionality of the representation space"""
        return len(self.tokenizer.get_vocab())

    def __call__(self, x: str) -> np.ndarray:
        """Represents x as a D-dimensional vector of real numbers"""
        unique_token_ids = list(set(self.tokenizer.encode(x).ids))

        # A 0 vector with the same length as the vocabulary except in positions
        # corresponding to tokens appearing in the document.
        vector = np.zeros(len(self))
        vector[unique_token_ids] = 1

        return vector


class BagOfWords:
    """Counts occurrences of each vocabulary token."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def __call__(self, text):
        vector = np.zeros(len(self))
        for token_id in self.tokenizer.encode(text).ids:
            vector[token_id] += 1

        return vector


class TFIDF:
    """Represents text using token counts weighted by document rarity."""

    def __init__(self, tokenizer, training_texts):
        self.tokenizer = tokenizer

        # Calculate the IDF values using the training texts.
        document_counts = np.zeros(len(self))

        for text in training_texts:
            # Each token contributes at most once per document.
            token_ids = set(self.tokenizer.encode(text).ids)
            for token_id in token_ids:
                document_counts[token_id] += 1

        # Tokens appearing in fewer documents receive higher IDF weights.
        # Adding 1 inside the fraction prevents division by zero.
        # Adding 1 after the logarithm keeps even common tokens' weights >= 1.
        self.idf = np.log((1 + len(training_texts)) / (1 + document_counts)) + 1

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def __call__(self, text):
        vector = np.zeros(len(self))

        # Count each token's occurrences in this text.
        for token_id in self.tokenizer.encode(text).ids:
            vector[token_id] += 1

        # Multiply each token's count by its corresponding IDF weight.
        vector *= self.idf

        # Scale nonzero vectors to length 1.
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm

        return vector
