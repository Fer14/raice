from race import Race
from neat_.neat_car import NeatCar
from policy_gradient.pg_car import PGCar
from qlearning.q_car import QCar
from sarsa.sarsa_car import SARSACar


def main():

    race = Race(
        start=[1295, 966],
        finnish_line=[(1230, 927), (1292, 927), (1304, 1024), (1236, 1030)],
    )

    car = NeatCar(position=race.start)
    car.load_net()

    car2 = PGCar(position=race.start)
    car2.load_policy()

    car3 = QCar(position=race.start)
    car3.load()
    car3.epsilon = 0.1

    car4 = SARSACar(position=race.start)
    car4.load()
    car4.epsilon = 0.1

    race.race(cars=[car, car2, car3, car4])

    # race.race([car4])


if __name__ == "__main__":
    main()
