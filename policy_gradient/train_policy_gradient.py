import torch
from pg_car import PGCar
from pg_race import PGRace
import pygame


def main():

    episodes = 200000
    train_every = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Detected device: ", device)

    race = PGRace(
        start=[1295, 966]
    )
    car = PGCar(position=race.start, device=device)

    try:
        race.training_race(car, episodes=episodes, train_every=train_every)
    except KeyboardInterrupt:
        print("Training interrupted. Saving current car...")
        car.save_policy()


if __name__ == "__main__":
    main()
