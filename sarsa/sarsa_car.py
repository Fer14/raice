import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import pygame
from collections import deque
import random

# from prioritized_replay_buffer import PrioritizedReplayBuffer

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

MIN_EPSILON = 0.1
EPSILON_DECAY = 0.9999


class SARSACar(Car):

    def __init__(
        self,
        position=None,
        device="cuda",
        input_size=5,
        hidden_size=5,
        output_size=4,
        discount_factor=0.99,
        learning_rate=5e-3,  # 1e-4,
    ):
        super().__init__(position=position, angle=0)
        self.device = device
        self.model = self.create_model(input_size, hidden_size, output_size).to(
            self.device
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.discount_factor = discount_factor
        self.epsilon = 1
        self.output_size = output_size

        self.sprite = pygame.image.load("./sarsa/car.png").convert_alpha()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        self.crashed = False

    def create_model(self, input_size, hidden_size, output_size, trainable=True):
        model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        if not trainable:
            for param in model.parameters():
                param.requires_grad = False

        return model

    def get_qs(self, state):
        return self.model(state).cpu().detach().numpy()

    def act_epsilon_greedy(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.output_size)
        else:
            return int(np.argmax(self.get_qs(state)))

    def act_race(self, state):
        state = torch.tensor(np.array(state), dtype=torch.float32).to(self.device)
        return int(np.argmax(self.get_qs(state)))

    def epsilon_decay(self):
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY
            self.epsilon = max(MIN_EPSILON, self.epsilon)

    def save(self):
        torch.save(self.model.state_dict(), "./sarsa/sarsa_policy.pth")

    def load(self):
        self.model.load_state_dict(torch.load("./sarsa/best.pth"))

    def train(self, state, action, reward, new_state, done):

        # Convert to PyTorch tensors
        state_tensor = (
            torch.tensor(np.array(state), dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )  # Add batch dimension
        new_state_tensor = (
            torch.tensor(np.array(new_state), dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )  # Add batch dimension

        # Get the Q-value for the current state-action pair
        current_q_values = self.model(state_tensor)  # Shape: [1, output_size]
        current_q_value = current_q_values.gather(
            1, torch.tensor([[action]], dtype=torch.long).to(self.device)
        )  # Shape: [1, 1]
        current_q_value = current_q_value.squeeze(1)  # Shape: [1]

        # If the episode is done, next_q_value should be 0
        if done:
            next_q_value = torch.tensor([0.0], dtype=torch.float32).to(self.device)
        else:
            # Get the next action using the epsilon-greedy policy for SARSA
            next_action = self.act_epsilon_greedy(new_state)
            next_q_values = self.model(new_state_tensor)  # Shape: [1, output_size]
            next_q_value = next_q_values.gather(
                1, torch.tensor([[next_action]], dtype=torch.long).to(self.device)
            )  # Shape: [1, 1]
            next_q_value = next_q_value.squeeze(1)  # Shape: [1]

        # Compute the target Q-value using the SARSA update rule
        target_q_value = (
            torch.tensor([reward], dtype=torch.float32).to(self.device)
            + self.discount_factor * next_q_value
        )

        # Calculate the loss between the current Q-value and target Q-value
        loss = nn.MSELoss()(current_q_value, target_q_value)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Decay epsilon for exploration-exploitation tradeoff
        self.epsilon_decay()

        return loss.item()

    def action_train(self, state):

        action = self.act_epsilon_greedy(state)

        if action == 0:
            self.angle += 10  # Left
        elif action == 1:
            self.angle -= 10  # Right
        elif action == 2:
            if self.speed - 2 >= 6:
                self.speed -= 2  # Slow Down
        else:
            if self.speed + 2 <= 10:
                self.speed += 2  # Speed Up

        return action

    def action(self):

        state = self.get_data()

        action = self.act_epsilon_greedy(state)

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

    def get_reward(self):
        if self.crashed:
            self.crashed = False
            return -100

        # Calculate reward based on distance and velocity
        # distance_traveled = np.linalg.norm(
        #     np.array(self.position) - np.array(self.last_position)
        # )
        distance_reward = self.distance / (CAR_SIZE_X / 2)
        velocity_reward = self.speed

        # total_reward = (
        #     0.1 * distance_reward + 0.1 * velocity_reward + 0.8 * distance_traveled
        # )

        total_reward = 0.7 * distance_reward + 0.3 * velocity_reward

        return total_reward
