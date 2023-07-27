import torch
from torch import nn

import numpy as np

class GestureRec(nn.Module):
    """
    The class for the Gesture Recognition model.
    """
    def __init__(self, input_size=14, hidden_size=256, output_size=5, lstm_size=1, fc_size=2, device=None):
        """
        The initializer method.
        :param input_size: The number of features in an input vector.
        :param hidden_size: The number of neurons in the hidden layers.
        :param output_size: The number of neurons in the output layers (number of classes).
        :param lstm_size: The size of the LSTM network (1 or 2).
        :param fc_size: The number of fully connected networks (1 or 2).
        :param device: The device where the code will run (cpu or cuda).
        """
        super(GestureRec, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm_layers = lstm_size
        self.fc_layers = fc_size
        self.device = device

        self.lstm1 = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=1,
                             batch_first=True)

        self.lstm2 = nn.LSTM(input_size=hidden_size,
                             hidden_size=hidden_size,
                             num_layers=1,
                             batch_first=True)

        self.seq = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hidden_0 = torch.zeros((1, x.size(0), self.hidden_size), dtype=torch.float32, device=self.device)
        cell_0 = torch.zeros((1, x.size(0), self.hidden_size), dtype=torch.float32, device=self.device)

        y1, (hidden_1, cell_1) = self.lstm1(x, (hidden_0, cell_0))
        if self.lstm_layers == 2:
            y1, (hidden_1, cell1) = self.lstm2(y1, (hidden_1, cell_1))

        hidden_1 = hidden_1.view(-1, self.hidden_size)

        if self.fc_layers == 2:
            hidden_1 = self.seq(hidden_1)

        y = self.linear(hidden_1)

        return y
