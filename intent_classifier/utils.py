"""
Utility Functions for training.

- word_to_index
- make_bow_vector
"""

import numpy as np


def word_to_index(vocab):
    """
    Encoding words by indexing.

    Parameters
    ----------
        vocab: list.

    Returns
    -------
        word_to_idx dictionary.
    """
    word_to_idx = {}

    for word in vocab:
        word_to_idx[word] += 1

    return word_to_idx


def make_bow_vector(word_seq, word_to_idx):
    """
    Make Bag of Words vector using word_to_idx dictionary.

    Parameters
    ----------
        word_seq: list.
        word_to_idx: dict.

    Returns
    -------
        bow_vec numpy array.
    """
    bow_vec = np.zeros(len(word_to_idx), dtype=float)

    for word in word_seq:
        bow_vec[word_to_idx[word]] = 1

    return bow_vec
