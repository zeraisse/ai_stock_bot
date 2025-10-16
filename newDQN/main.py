import os
import DQNAgent as dqn
import TradingEnv as te
import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


os.makedirs('./models', exist_ok=True)

data = pd.read_csv('../dataset/top10_stocks_2025.csv')
prices = data['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler()
prices = scaler.fit_transform(prices).flatten()
env = te.TradingEnv(prices)
agent = dqn.DQNAgent(state_size=3, action_size=3)

for episode in range(10):
    state = env.reset()
    state = np.reshape(state, [1, 3])
    total_reward = 0
    for time in range(len(prices)-1):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        next_state = np.reshape(next_state, [1, 3])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"Episode: {episode+1}/10, Total Reward: {total_reward}")
            break
    if len(agent.memory) > 32:
        agent.replay(32)
agent.save("models/dqn_trading_model.h5")
