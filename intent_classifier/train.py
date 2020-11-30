"""
Training the network.
"""
from utils import make_bow_vector
from preprocess import make_dataset, NeuralNetData
from model import NeuralNet

# torch imports.
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn


def train_model():
    """
    Trains the network.

    Yields
    -----
        learned_parameters.pth
    """

    vocab, word_to_idx, train_X, train_y = make_dataset()
    dataset = NeuralNetData(train_X, train_y)

    input_size = len(vocab)
    output_size = len(train_y)
    hidden_size = 8
    batch_size = 8
    learning_rate = 0.01
    num_epoch = 300
    torch.manual_seed(0)

    # Initializing.
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = NeuralNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop.
    for epoch in range(num_epoch):
        for instance, target in data_loader:
            output = model(instance)
            loss = loss_function(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"EPOCH:{epoch}, LOSS: {loss.item()}")

    # Saving Learned Parameters.
    learned_parameters = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "vocab": vocab,
        "word_to_idx": word_to_idx
    }

    torch.save(learned_parameters, "learned_parameters.pth")
    return
