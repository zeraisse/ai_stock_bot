import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class DQNModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 24),
            nn.ReLU(),                 
            nn.Linear(24, 24),         
            nn.ReLU(),
            nn.Linear(24, action_size) 
        )

    def forward(self, x):
        return self.net(x)
