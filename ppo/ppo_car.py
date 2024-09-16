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


class PPOCar(Car):

    def __init__(
        self,
        position=None,
        device="cuda",
        input_size=5,
        hidden_size=5,
        output_size=4,
        discount_factor=0.99,
        learning_rate=1e-3,  # best is 1e-3
        epsilon=0.2,  # PPO clipping parameter
        entropy_coef=0.01,
    ):
        super().__init__(position=position, angle=0)
        self.device = device
        self.model = self.create_model(input_size, hidden_size, output_size).to(
            self.device
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef

        self.sprite = pygame.image.load("./ppo/car.png").convert_alpha()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        self.onpolicy_reset()
        self.crashed = False
        self.name = "PPO"

    def create_model(self, input_size, hidden_size, output_size):
        model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1),
        )
        return model

    def save_policy(self):
        torch.save(self.model.state_dict(), "./ppo/policy.pth")

    def load_policy(self):
        self.model.load_state_dict(torch.load("./ppo/policy_best.pth"))

    def onpolicy_reset(self):
        self.states = []
        self.actions = []
        self.log_probs = []

    def forward(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs = self.model(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)

        return action.item()

    def compute_returns(self, rewards):
        R = 0
        returns = []
        for r in rewards[::-1]:
            R = r + self.discount_factor * R
            returns.insert(0, R)
        returns = torch.tensor(returns).float().to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def train(self, rewards):
        returns = self.compute_returns(rewards)
        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        old_log_probs = torch.cat(self.log_probs).detach()

        for _ in range(5):  # PPO update iterations
            action_probs = self.model(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * returns
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * returns

            loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.onpolicy_reset()
        return loss.item()

    def action_train(self, state):

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

        return action

    def action(self):
        state = self.get_data()
        action = self.forward(state)

        if action == 0:
            self.angle += 10  # Left
            self.n_drifts_left += 1
            self.n_drifts_right = 0
        elif action == 1:
            self.angle -= 10  # Right
            self.n_drifts_left = 0
            self.n_drifts_right += 1
        elif action == 2:
            self.n_drifts_right = 0
            self.n_drifts_left = 0
            if self.speed - 2 >= 6:
                self.speed -= 2  # Slow Down
        else:
            self.n_drifts_right = 0
            self.n_drifts_left = 0
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
