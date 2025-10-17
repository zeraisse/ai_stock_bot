import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class DQNModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNModel, self).__init__()
        self.net = nn.LSTM(state_size, 128, num_layers=2, batch_first=True)
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size) 
        )

    def forward(self, x):
        return self.net(x)
