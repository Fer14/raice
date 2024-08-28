import pygame
from race import Race
from neat_car import NeatCar


class NeatRace(Race):

    def training_race(self, cars: list[NeatCar], genomes):
        clock = pygame.time.Clock()

        counter = 0

        running = True
        while running:
            clock.tick(120)  # Cap the frame rate

            # Exit On Quit Event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # throw exception to stop the race
                    raise KeyboardInterrupt("Race interrupted")

            # For Each Car Get The Acton It Takes
            for car in cars:
                car.action()
            # Check If Car Is Still Alive
            # Increase Fitness If Yes And Break Loop If Not
            still_alive = 0
            for i, car in enumerate(cars):
                if car.is_alive():
                    still_alive += 1
                    car.update(self.game_map)
                    genomes[i][1].fitness += car.get_reward()

            if still_alive == 0:
                break

            counter += 1
            if counter == 30 * 40:  # Stop After About 20 Seconds
                break

            self.draw(cars)
