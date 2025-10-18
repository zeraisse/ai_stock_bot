import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import DQNLSTMModel as model
import DQNTFTModel as tft_model

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999
        self.learning_rate = 0.0001
        self.learning_rate = 0.0005  # adam % apprentissage

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.model = model.DQNModel(state_size, action_size).to(self.device)
        self.model = model.DQNLSTMModel(state_size, action_size).to(self.device)
        self.model = tft_model.DQNTFTModel(state_size, action_size).to(self.device) # ici CHAMPION j'utilie le TFT models

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).to(self.device)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)

            q_values = self.model(state)

            with torch.no_grad():
                next_q = self.model(next_state)
                target = reward if done else reward + self.gamma * torch.max(next_q).item()

            target_f = q_values.clone().detach()
            target_f[0][action] = target

            self.optimizer.zero_grad()
            loss = self.loss_fn(q_values, target_f)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
