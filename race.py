import pygame
import sys
import random
import os
from car import Car

WIDTH = 1920
HEIGHT = 1080


class Race:

    def __init__(self, start: list[int, int]) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))  # , pygame.FULLSCREEN)
        pygame.display.set_caption("RAICE")
        self.game_map = pygame.image.load("./maps/bahrain2.png").convert()
        self.generation_font = pygame.font.SysFont("Arial", 30)
        self.alive_font = pygame.font.SysFont("Arial", 20)
        self.start = start

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
                car.update(self.game_map)

            self.draw(cars)

        print("Race finished")
