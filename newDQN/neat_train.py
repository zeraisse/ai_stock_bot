import os
import pickle
from typing import Tuple, List, Callable, Any

import neat
import neat.reporting as reporting
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


def run_neat(
    config_file: str,
    generations: int = 500,
    reporters: List[Any] | None = None,
) -> Tuple[object, neat.StatisticsReporter, list]:
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
    
    class SimpleLogger(reporting.BaseReporter):
        def start_generation(self, generation):
            try:
                print(f"[NEAT] Generation {generation}", flush=True)
            except Exception:
                pass
        def post_evaluate(self, config, population, species, best_genome):
            try:
                best_fit = getattr(best_genome, 'fitness', None)
                if best_fit is not None:
                    print(f"[NEAT] Best fitness: {best_fit:.3f}", flush=True)
            except Exception:
                pass

    pop.add_reporter(SimpleLogger())

    # Attach custom reporters if provided
    if reporters:
        for rep in reporters:
            try:
                # If user provided a callable object with hooks, adapt it
                if not hasattr(rep, 'start_generation') and callable(rep):
                    class _Wrapper(reporting.BaseReporter):
                        def __init__(self, fn: Callable):
                            self.fn = fn
                        def start_generation(self, generation):
                            try:
                                self.fn('start_generation', generation)
                            except Exception:
                                pass
                        def post_evaluate(self, config, population, species, best_genome):
                            try:
                                self.fn('post_evaluate', best_genome)
                            except Exception:
                                pass
                    pop.add_reporter(_Wrapper(rep))
                else:
                    pop.add_reporter(rep)
            except Exception:
                pass

    winner = pop.run(eval_genomes, n=generations)

    os.makedirs('./models', exist_ok=True)
    with open('./models/best_genome.pkl', 'wb') as f:
        pickle.dump(winner, f)

    # Build per-generation best fitness history
    try:
        fitness_history = [float(g.fitness) for g in getattr(stats, 'most_fit_genomes', [])]
    except Exception:
        fitness_history = []
    return winner, stats, fitness_history


if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), 'neat-config.ini')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"NEAT config not found at {config_path}")
    winner_genome, stats_reporter, fitness_history = run_neat(config_path, generations=30)
    print("NEAT training done.")


