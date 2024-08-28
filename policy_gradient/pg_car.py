import sys
import torch.optim as optim
import torch.nn as nn
import torch
from torch.distributions import Categorical
import numpy as np
import pygame

# Add this at the top of the file
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


from car import Car


CAR_SIZE_X = 30
CAR_SIZE_Y = 30

WIDTH = 1920
HEIGHT = 1080

BORDER_COLOR = (255, 255, 255, 255)  # Color To Crash on Hit


class PGCar(Car):

    def __init__(
        self,
        position=None,
        device="gpu",
        input_size=5,
        hidden_size=5,
        output_size=4,
        discount_factor=0.99,
        learning_rate=1e-4,  # best is 1e-3
    ):
        super().__init__(position=position, angle=0)
        self.device = device
        self.model = self.create_model(input_size, hidden_size, output_size).to(
            self.device
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.discount_factor = discount_factor

        self.sprite = pygame.image.load("./policy_gradient/car.png").convert()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        self.onpolicy_reset()
        self.crashed = False

    def create_model(self, input_size, hidden_size, output_size):
        model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1),
        )
        return model

    def save_policy(self):
        torch.save(self.model.state_dict(), "./policy_gradient/policy.pth")

    # def load_policy(self):
    #     self.model.load_state_dict(torch.load("policy.pth"))

    def onpolicy_reset(self):
        self.log_probs = []

    def forward(self, state):
        state = np.array(state, dtype=np.float32)  # Convert list to numpy array
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.model(state)
        m = Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return action.item()

    def train(self, rewards):
        self.optimizer.zero_grad()

        returns = []
        future_return = 0
        for r in reversed(rewards):
            future_return = r + self.discount_factor * future_return
            returns.insert(0, future_return)

        returns = torch.tensor(returns).to(self.device)

        # Add a check for non-zero standard deviation
        if returns.std() > 1e-6:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        else:
            returns = returns - returns.mean()

        policy_loss = []

        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)

        policy_loss = torch.stack(policy_loss).sum()

        policy_loss.backward()
        self.optimizer.step()
        self.onpolicy_reset()
        return policy_loss.item()

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            # If Any Corner Touches Border Color -> Crash
            # Assumes Rectangle
            if (
                point[0] < 0 or point[0] > WIDTH or point[1] < 0 or point[1] > HEIGHT
            ) or (game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR):
                self.alive = False
                self.crashed = True
                self.reset()
                break

        # Check if the car is crossing the starting line from right to left
        # if (
        #     self.position[0] < self.start_position[0]
        #     and self.angle > 90
        #     and self.angle < 270
        # ):
        #     self.alive = False
        #     self.crashed = True
        #     self.reset()

    def reset(self):
        self.position = self.start_position
        self.speed = 0
        self.angle = 0
        self.distance = 0
        # self.alive = True

    def action(self, state):

        action = self.forward(state)

        if action == 0:
            self.angle += 10  # Left
        elif action == 1:
            self.angle -= 10  # Right
        elif action == 2:
            if self.speed - 2 >= 6:
                self.speed -= 2  # Slow Down
        else:
            self.speed += 2  # Speed Up

    def get_reward(self):
        if self.crashed:
            self.crashed = False
            return -100

        # Calculate reward based on distance and velocity
        distance_reward = self.distance / (CAR_SIZE_X / 2)
        velocity_reward = self.speed  # Assuming max speed is 20, adjust as needed

        # Combine the rewards (you can adjust the weights)
        total_reward = 0.7 * distance_reward + 0.3 * velocity_reward

        return total_reward
