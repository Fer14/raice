import pygame
import sys
import random
import os
from car import Car

WIDTH = 1920
HEIGHT = 1080
CAR_SIZE_X = 30
CAR_SIZE_Y = 30


class Race:

    def __init__(self, start: list[int, int], finnish_line) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))  # , pygame.FULLSCREEN)
        pygame.display.set_caption("RAICE")
        self.game_map = pygame.image.load("./maps/bahrain2.png").convert()
        self.generation_font = pygame.font.SysFont("Arial", 30)
        self.alive_font = pygame.font.SysFont("Arial", 20)
        self.start = start
        x_coords = [point[0] for point in finnish_line]
        y_coords = [point[1] for point in finnish_line]

        # Compute the bounding box
        left = min(x_coords)
        top = min(y_coords)
        right = max(x_coords)
        bottom = max(y_coords)
        # Compute width and height
        width = right - left
        height = bottom - top

        self.finnish_line = pygame.Rect(left, top, width, height)

        self.crash = pygame.image.load("./images/smoke.png").convert_alpha()
        self.crash = pygame.transform.scale(self.crash, (50, 50))

    def load_random_map(self):
        map_dir = "../maps"  # Directory containing map images
        map_files = [f for f in os.listdir(map_dir) if f.endswith(".png")]
        if not map_files:
            raise FileNotFoundError("No map images found in the maps directory")
        random_map = random.choice(map_files)
        self.game_map = pygame.image.load(os.path.join(map_dir, random_map)).convert()

    def draw(self, cars: list[Car], draw_radar=False):
        self.screen.blit(self.game_map, (0, 0))

        for car in cars:
            if car.is_alive():
                car.draw(self.screen, draw_radar)

            if car.crashed:
                # self.crashes.append(self.screen.blit(self.crash, car.crashed_position))
                self.screen.blit(self.crash, car.crashed_position)

        # pygame.draw.rect(self.screen, (255, 0, 0), self.finnish_line, 2)

        pygame.display.flip()

    def race(self, cars: list[Car]):
        clock = pygame.time.Clock()

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
                car.update(self.game_map, training=False)
                # self.check_lap(car)

            self.draw(cars)

        print("Race finished")

    def check_lap(self, car: Car):
        car_rect = pygame.Rect(car.position[0], car.position[1], CAR_SIZE_X, CAR_SIZE_Y)
        if self.finnish_line.contains(car_rect):
            car.laps += 1
            print(f"Lap completed! Total laps: {car.laps}")
