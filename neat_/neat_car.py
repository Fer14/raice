import pygame
import math
import os
import sys
import neat
import pickle

# Add this at the top of the file
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


from car import Car


CAR_SIZE_X = 30
CAR_SIZE_Y = 30

BORDER_COLOR = (255, 255, 255, 255)  # Color To Crash on Hit


class NeatCar(Car):

    def __init__(self, net=None, position=None):
        super().__init__(position=position, angle=0)
        self.net = net
        self.sprite = pygame.image.load("./neat_/car.png").convert_alpha()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

    def load_net(self):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, "config.txt")
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )
        with open("./neat_/checkpoints/2024-08-25/best_genome.pickle", "rb") as f:
            genome = pickle.load(f)
            self.net = neat.nn.FeedForwardNetwork.create(genome, config)

    def get_reward(self):
        # Calculate reward based on distance and velocity
        distance_reward = self.distance / (CAR_SIZE_X / 2)
        velocity_reward = self.speed / 20  # Assuming max speed is 20, adjust as needed

        # Combine the rewards (you can adjust the weights)
        total_reward = 0.7 * distance_reward + 0.3 * velocity_reward

        return total_reward
        # return  self.distance / (CAR_SIZE_X / 2)

    def action(self):
        input = self.get_data()
        output = self.net.activate(input)
        choice = output.index(max(output))

        if choice == 0:
            self.angle += 10  # Left
        elif choice == 1:
            self.angle -= 10  # Right
        elif choice == 2:
            if self.speed - 2 >= 6:
                self.speed -= 2  # Slow Down
        else:
            self.speed += 2  # Speed Up
