import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.lstm.lstm_predictor import LSTMPredictor
from models.tft.tft_predictor import TFTPredictor

class DQNLSTMModel(nn.Module):
    def __init__(self, state_size, action_size, sequence_length=60):
        super(DQNLSTMModel, self).__init__()

        self.lstm_predictor = LSTMPredictor(
            sequence_length=sequence_length,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # LSTM interne
        self.lstm = self.lstm_predictor.build_model(input_size=state_size)

        
        self.fc_q = nn.Linear(self.lstm_predictor.hidden_size, action_size)

    def forward(self, x):
        
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        out, _ = self.lstm.lstm(x) 
        out = out[:, -1, :] 
        q_values = self.fc_q(out)
        return q_values
