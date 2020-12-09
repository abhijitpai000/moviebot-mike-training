"""
Pre-process data.json.

"""
from nltk.tokenize import word_tokenize
import json

# PyTorch imports.
import torch
from torch.utils.data import Dataset


# Local Imports.
from intent_classifier.utils import word_to_index, make_bow_vector


def _tokenize_input(input_seq):
    """
    Tokenize input sequence.

    Parameters
    ----------
        input_seq: list.

    Returns
    -------
        tokenized words list.
    """
    tokens = word_tokenize(input_seq)
    return [w.lower() for w in tokens]


def make_dataset():
    """
    Extract data from raw file for training.

    Returns
    -------
        vocab, word_to_idx, intents, train_X, train_y
    """
    with open("intent_classifier/datasets/data.json", "r") as f:
        data = json.load(f)

    vocab = []
    training_data = []
    intents = []

    for i in data["intents"]:
        tag = i["intent"]
        intents.append(tag)
        for j in i["question"]:
            words = _tokenize_input(j)
            vocab.extend(words)
            training_data.append((words, tag))

    vocab = set(vocab)
    word_to_idx = word_to_index(vocab)

    train_X = []
    train_y = []

    for instance, label in training_data:
        bow_vec = make_bow_vector(instance, word_to_idx)
        train_X.append(bow_vec)

        target = intents.index(label)
        train_y.append(target)

    train_X = torch.FloatTensor(train_X)
    train_y = torch.LongTensor(train_y)

    return vocab, word_to_idx, intents, train_X, train_y


class NeuralNetData(Dataset):
    def __init__(self, train_X, train_y):
        """
        Parameters
        ----------
            train_X (tensor) : instances.
            train_y (tensor) : label.
        """
        self.train_X = train_X
        self.train_y = train_y
        self.n_samples = len(train_X)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.train_X[idx], self.train_y[idx]