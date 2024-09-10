import pygame
import sys
import random
import os
from car import Car
import time

WIDTH = 1920
HEIGHT = 1080
CAR_SIZE_X = 30
CAR_SIZE_Y = 30


class Race:

    def __init__(self, start: list[int, int], finnish_line, laps: int = 2) -> None:
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

        self.laps = laps
        self.finished_cars = []
        self.logo = pygame.image.load("./logos/small.png").convert()
        self.logo = pygame.transform.scale(self.logo, (250, 60))

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
        self.draw_standings_table(cars + self.finished_cars)
        pygame.display.flip()

    def race(self, cars: list[Car]):
        clock = pygame.time.Clock()

        time_init = time.time()

        for car in cars:
            car.init_time(time_init)

        running = True
        while running:
            clock.tick(120)  # Cap the frame rate

            # Exit On Quit Event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # throw exception to stop the race
                    raise KeyboardInterrupt("Race interrupted")

            # For Each Car Get The Acton It Takes
            for car in cars[:]:
                car.action()
                car.update(self.game_map, training=False)
                self.check_lap(car)

                if car.laps == self.laps:
                    print(f"Car has completed all laps: {car.laps}")
                    self.finished_cars.append(car)
                    cars.remove(car)

            self.draw(cars)

        print("Race finished")

    def check_lap(self, car: Car):
        car_rect = pygame.Rect(car.position[0], car.position[1], CAR_SIZE_X, CAR_SIZE_Y)
        # Check if the car is intersecting with the finish line
        if self.finnish_line.colliderect(car_rect):
            # Calculate the direction the car is moving
            dx = car.position[0] - car.last_positions[car.len_positions - 2][0]

            # Determine if the car is crossing in the correct direction
            # Adjust these conditions based on your track layout
            correct_direction = (
                dx > 0
            )  # Assuming the car should cross from left to right

            # Check if the car wasn't on the finish line in the last frame
            if not hasattr(car, "on_finish_line") or not car.on_finish_line:
                if correct_direction:
                    car.laps += 1
                    car.lap_times.append(time.time() - car.init_time_)

                # Mark that the car is on the finish line
                car.on_finish_line = True
        else:
            # Car is not on the finish line
            car.on_finish_line = False

    def draw_standings_table(self, cars: list[Car]):
        # Sort cars by laps (descending) and then by distance (descending)
        sorted_cars = sorted(
            cars,
            key=lambda car: (
                -car.laps,
                car.lap_times[-1] if car.lap_times else float("inf"),
            ),
        )
        # Table settings
        logo_height = self.logo.get_height()
        table_width = 300
        header_height = 60
        row_height = 30
        table_height = min(
            HEIGHT - 40, header_height + len(cars) * row_height
        )  # Adjust based on number of cars
        table_x = WIDTH - table_width - 20
        table_y = 20 + logo_height

        logo_y = 10  # Padding from the top of the screen
        self.screen.blit(self.logo, (table_x + 20, logo_y))

        # Draw table background
        pygame.draw.rect(
            self.screen, (17, 75, 95), (table_x, table_y, table_width, table_height)
        )
        pygame.draw.rect(
            self.screen, (0, 0, 0), (table_x, table_y, table_width, table_height), 2
        )

        # Draw header
        header_font = pygame.font.SysFont("Arial", 20, bold=True)
        header = header_font.render("Race Standings", True, (255, 255, 255))
        header_2 = header_font.render(f"Total laps: {self.laps}", True, (255, 255, 255))
        self.screen.blit(header, (table_x + 10, table_y + 5))
        self.screen.blit(header_2, (table_x + 180, table_y + 5))

        # Draw column titles
        column_font = pygame.font.SysFont("Arial", 16, bold=True)
        pos_title = column_font.render("Pos", True, (255, 255, 255))
        name_title = column_font.render("Name", True, (255, 255, 255))
        laps_title = column_font.render("Laps", True, (255, 255, 255))
        dist_title = column_font.render("Lap time", True, (255, 255, 255))
        self.screen.blit(pos_title, (table_x + 10, table_y + 35))
        self.screen.blit(name_title, (table_x + 60, table_y + 35))
        self.screen.blit(laps_title, (table_x + 140, table_y + 35))
        self.screen.blit(dist_title, (table_x + 200, table_y + 35))

        # Draw standings
        font = pygame.font.SysFont("Arial", 16)
        for i, car in enumerate(sorted_cars):
            y = table_y + (i + 2) * row_height
            if y + row_height > table_y + table_height:
                break  # Stop if we run out of space in the table

            # Draw alternating row backgrounds
            if i % 2 == 0:
                pygame.draw.rect(
                    self.screen, (17, 75, 95), (table_x, y, table_width, row_height)
                )
            else:
                pygame.draw.rect(
                    self.screen, (26, 147, 111), (table_x, y, table_width, row_height)
                )

            if car.lap_times:
                last_lap_time = car.lap_times[-1]
            else:
                last_lap_time = 0

            pos_text = font.render(f"{i + 1}", True, (255, 255, 255))
            name_text = font.render(f"{car.name}", True, (255, 255, 255))
            laps_text = font.render(f"{car.laps}", True, (255, 255, 255))
            dist_text = font.render(f"{last_lap_time:.2f}", True, (255, 255, 255))

            self.screen.blit(pos_text, (table_x + 10, y + 5))
            self.screen.blit(name_text, (table_x + 60, y + 5))
            self.screen.blit(laps_text, (table_x + 140, y + 5))
            self.screen.blit(dist_text, (table_x + 200, y + 5))

        # Redraw the border to clean up any overflow
        pygame.draw.rect(
            self.screen, (0, 0, 0), (table_x, table_y, table_width, table_height), 2
        )
