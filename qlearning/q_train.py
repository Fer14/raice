import torch
from q_car import QCar
from q_race import QRace


def main():

    episodes = 200000
    train_every = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Detected device: ", device)

    race = QRace(start=[1295, 966])
    car = QCar(position=race.start, device=device)

    try:
        race.training_race(car, episodes=episodes, train_every=train_every)
    except KeyboardInterrupt:
        print("Training interrupted. Saving current car...")
        car.save()


if __name__ == "__main__":
    main()
