import os
import neat
from datetime import date
from neat_car import NeatCar
from neat_race import NeatRace

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from race import Race


race = NeatRace(start=(1295, 966))


def eval_genomes(genomes, config, shuffle=False):

    cars = []

    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        g.fitness = 0
        cars.append(NeatCar(net=net, position=race.start))

    # race.load_random_map()
    race.training_race(cars, genomes)


def run_neat(config):
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    try:
        winner = p.run(eval_genomes, 200)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current best genome...")
        winner = p.best_genome
    finally:
        # Save the winner genome
        # with open(f"checkpoints/{date.today()}/best_genome.pickle", "wb") as f:
        #     pickle.dump(winner, f)
        print(f"Best genome saved to checkpoints/{date.today()}/best_genome.pickle")


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    os.makedirs(f"checkpoints/{date.today()}", exist_ok=True)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    run_neat(config)
