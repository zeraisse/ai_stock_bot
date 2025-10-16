import os
import pickle
from typing import Tuple

import neat
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import TradingEnv as te


def make_env():
    data = pd.read_csv('../dataset/top10_stocks_2025.csv')
    prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    prices = scaler.fit_transform(prices).flatten()
    return te.TradingEnv(prices)


def eval_genome(genome, config) -> float:
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = make_env()
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    done = False
    total_reward = 0.0
    while not done:
        outputs = net.activate(obs.astype(np.float32).tolist())
        action = int(np.argmax(outputs))
        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, _ = step_out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, _ = step_out
        total_reward += float(reward)
    # Option alternative: fitness = net worth final
    return total_reward


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run_neat(config_file: str, generations: int = 30) -> Tuple[object, neat.StatisticsReporter]:
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    winner = pop.run(eval_genomes, n=generations)

    os.makedirs('./models', exist_ok=True)
    with open('./models/best_genome.pkl', 'wb') as f:
        pickle.dump(winner, f)

    return winner, stats


if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), 'neat-config.ini')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"NEAT config not found at {config_path}")
    winner_genome, stats_reporter = run_neat(config_path, generations=30)
    print("NEAT training done.")


