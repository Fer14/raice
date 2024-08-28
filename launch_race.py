from race import Race
from neat_.neat_car import NeatCar


def main():

    race = Race(
        start=[1295, 966],
    )

    car = NeatCar(position=race.start)
    car.load_net()

    race.race(cars=[car])


if __name__ == "__main__":
    main()
