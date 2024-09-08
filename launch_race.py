from race import Race
from neat_.neat_car import NeatCar
from policy_gradient.pg_car import PGCar
from qlearning.q_car import QCar


def main():

    race = Race(
        start=[1295, 966],
    )

    car = NeatCar(position=race.start)
    car.load_net()

    car2 = PGCar(position=race.start)
    car2.load_policy()

    car3 = QCar(position=race.start)
    car3.load()

    race.race(cars=[car, car2, car3])


if __name__ == "__main__":
    main()
