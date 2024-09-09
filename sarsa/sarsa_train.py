import torch
from sarsa_car import SARSACar
from sarsa_race import SARSARace


def main():

    episodes = 200000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Detected device: ", device)

    race = SARSARace(start=[1295, 966])
    car = SARSACar(position=race.start, device=device)

    try:
        race.training_race(car, episodes=episodes)
    except KeyboardInterrupt:
        print("Training interrupted. Saving current car...")
        car.save()


if __name__ == "__main__":
    main()
