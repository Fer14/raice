import torch
from ppo_car import PPOCar
from ppo_race import PPORace


def main():

    episodes = 200000
    train_every = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Detected device: ", device)

    race = PPORace(
        start=[1295, 966],
        finnish_line=[(1230, 927), (1292, 927), (1304, 1024), (1236, 1030)],
    )
    car = PPOCar(position=race.start, device=device)

    try:
        race.training_race(car, episodes=episodes, train_every=train_every)
    except KeyboardInterrupt:
        print("Training interrupted. Saving current car...")
        car.save_policy()


if __name__ == "__main__":
    main()
