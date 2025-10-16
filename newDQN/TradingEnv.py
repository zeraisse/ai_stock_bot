import numpy as np
import gym
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, price):
        super().__init__()
        self.price = price
        self.current_step = 0
        self.balance = 1000
        self.holding = 0

        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def reset(self):
        self.current_step = 0
        self.balance = 1000
        self.holding = 0
        return self._get_obs()

    def _get_obs(self):
        return np.array([self.balance, self.holding, self.price[self.current_step]], dtype=np.float32)

    def step(self, action):
        current_price = self.price[self.current_step]
        reward = 0

        if action == 1:
            if self.balance >= current_price:
                self.holding += 1
                self.balance -= current_price
        elif action == 2:
            if self.holding > 0:
                self.holding -= 1
                self.balance += current_price

        self.current_step += 1
        done = self.current_step >= len(self.price) - 1

        new_value = self.balance + self.holding * current_price
        reward = new_value - (self.balance + self.holding * current_price)

        return self._get_obs(), reward, done, {}