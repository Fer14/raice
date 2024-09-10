import pygame
import math
import numpy as np
from collections import deque
import time

WIDTH = 1920
HEIGHT = 1080

CAR_SIZE_X = 30
CAR_SIZE_Y = 30

BORDER_COLOR = (255, 255, 255, 255)  # Color To Crash on Hit


class Car:

    def __init__(self, position, angle=0, len_positions=20):

        # self.position = [690, 740] # Starting Position
        self.position = position
        self.start_position = position
        self.angle = angle
        self.speed = 0

        self.speed_set = False  # Flag For Default Speed Later on

        self.center = [
            self.position[0] + CAR_SIZE_X / 2,
            self.position[1] + CAR_SIZE_Y / 2,
        ]  # Calculate Center

        self.radars = []  # List For Sensors / Radars
        self.drawing_radars = []  # Radars To Be Drawn

        self.alive = True  # Boolean To Check If Car is Crashed

        self.distance = 0  # Distance Driven
        self.time = 0  # Time Passed
        self.laps = 0
        self.last_position = self.position
        self.last_positions = deque(maxlen=len_positions)
        self.last_angles = deque(maxlen=len_positions)
        self.crashed = False
        self.n_drifts_left = 0
        self.n_drifts_right = 0
        self.len_positions = len_positions

        drift = pygame.image.load("./images/drift.png").convert_alpha()
        self.drift = pygame.transform.scale(drift, (20, 40))
        self.lap_times = []

    def __str__(self):
        return f"{self.name}"

    def init_time(self, time):
        self.init_time_ = time

    def draw(self, screen, draw_radar=False):
        if self.n_drifts_left >= 3 or self.n_drifts_right >= 3:
            if len(self.last_positions) >= self.len_positions:
                screen.blit(
                    pygame.transform.rotate(self.drift, 90 + self.angle),
                    self.last_positions[self.len_positions - 2],
                )
        screen.blit(self.rotated_sprite, self.position)  # Draw Sprite
        if draw_radar:
            self.draw_radar(screen)  # OPTIONAL FOR SENSORS

    def draw_radar(self, screen):
        # Optionally Draw All Sensors / Radars
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

    def check_collision(self, game_map, training):
        self.alive = True
        for point in self.corners:
            # If Any Corner Touches Border Color -> Crash
            # Assumes Rectangle
            if (
                point[0] < 0 or point[0] > WIDTH or point[1] < 0 or point[1] > HEIGHT
            ) or (game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR):
                self.crashed = True
                self.alive = False
                if training:
                    self.reset_training()
                else:
                    self.reset_race()

                break

    def reset_training(self):
        self.position = self.start_position
        self.speed = 0
        self.angle = 0
        self.distance = 0
        self.last_position = self.position
        self.laps = 0

    def reset_race(self):
        self.crashed_position = self.position
        self.position = self.last_positions[0]
        self.angle = self.last_angles[0]
        self.speed = 0
        self.last_position = self.position

    def check_radar(self, degree, game_map):
        length = 0
        x = int(
            self.center[0]
            + math.cos(math.radians(360 - (self.angle + degree))) * length
        )
        y = int(
            self.center[1]
            + math.sin(math.radians(360 - (self.angle + degree))) * length
        )

        # While We Don't Hit BORDER_COLOR AND length < 300 (just a max) -> go further and further
        while (
            not (x < 0 or x > WIDTH or y < 0 or y > HEIGHT)
            and not game_map.get_at((x, y)) == BORDER_COLOR
            and length < 300
        ):
            length = length + 1
            x = int(
                self.center[0]
                + math.cos(math.radians(360 - (self.angle + degree))) * length
            )
            y = int(
                self.center[1]
                + math.sin(math.radians(360 - (self.angle + degree))) * length
            )

        # Calculate Distance To Border And Append To Radars List
        dist = int(
            math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2))
        )
        self.radars.append([(x, y), dist])

    def update(self, game_map, training=True):

        # Set The Speed To 20 For The First Time
        # Only When Having 4 Output Nodes With Speed Up and Down
        if not self.speed_set:
            self.speed = 10
            self.speed_set = True

        # Get Rotated Sprite And Move Into The Right X-Direction
        # Don't Let The Car Go Closer Than 20px To The Edge
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)

        new_x = self.position[0] + math.cos(math.radians(360 - self.angle)) * self.speed
        new_x = min(max(new_x, 20), WIDTH - 120)

        # self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        # self.position[0] = max(self.position[0], 20)
        # self.position[0] = min(self.position[0], WIDTH - 120)

        # Same For Y-Position
        new_y = self.position[1] + math.sin(math.radians(360 - self.angle)) * self.speed
        new_y = min(max(new_y, 20), WIDTH - 120)

        # self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        # self.position[1] = max(self.position[1], 20)
        # self.position[1] = min(self.position[1], WIDTH - 120)

        self.position = (new_x, new_y)

        # Calculate New Center
        self.center = [
            int(self.position[0]) + CAR_SIZE_X / 2,
            int(self.position[1]) + CAR_SIZE_Y / 2,
        ]

        # Increase Distance and Time
        self.distance += self.speed
        self.time += 1

        # Calculate Four Corners
        # Length Is Half The Side
        length = 0.5 * CAR_SIZE_X
        left_top = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length,
        ]
        right_top = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length,
        ]
        left_bottom = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length,
        ]
        right_bottom = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length,
        ]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        # Check Collisions And Clear Radars
        self.check_collision(game_map, training)
        self.radars.clear()

        # From -90 To 120 With Step-Size 45 Check Radar
        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

        self.last_position = self.position
        self.last_angles.append(self.angle)
        self.last_positions.append(self.last_position)

    def get_data(self):
        # Get Distances To Border
        radars = self.radars
        return_values = [0, 0, 0, 0, 0]
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] / 30)

        return return_values

    def is_alive(self):
        # Basic Alive Function
        return self.alive

    def rotate_center(self, image, angle):
        # Rotate The Rectangle
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image

    def action(self):
        pass
