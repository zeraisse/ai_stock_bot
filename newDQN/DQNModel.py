import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class DQNModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNModel, self).__init__()
        
        self.hidden_size = 32
        
        # Couche LSTM
        self.lstm = nn.LSTM(state_size, self.hidden_size, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )


    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        
        return self.fc(x)
