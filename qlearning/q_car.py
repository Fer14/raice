import torch.optim as optim
import torch.nn as nn
import torch
from torch.distributions import Categorical
import numpy as np
import pygame
from collections import deque
import random

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

REPLAY_MEMORY_SIZE = 20000
MIN_REPLAY_MEMORY_SIZE = 1000
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.9995


class QCar(Car):

    def __init__(
        self,
        position=None,
        device="gpu",
        input_size=5,
        hidden_size=5,
        output_size=4,
        discount_factor=0.99,
        learning_rate=1e-3,
        mini_batch_size=32,
        update_target_every=100,
    ):
        super().__init__(position=position, angle=0)
        self.device = device
        self.model = self.create_model(input_size, hidden_size, output_size).to(
            self.device
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.target_model = self.create_model(
            input_size, hidden_size, output_size, trainable=False
        ).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

        self.discount_factor = discount_factor
        self.epsilon = 1
        self.mini_batch_size = mini_batch_size
        self.output_size = output_size
        self.update_target_every = update_target_every

        self.sprite = pygame.image.load("./qlearning/car.png").convert()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        self.crashed = False

    def create_model(self, input_size, hidden_size, output_size, trainable=True):
        model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1),
        )
        if not trainable:
            for param in model.parameters():
                param.requires_grad = False

        return model

    def update_target_network(self):
        self.target_update_counter += 1
        if self.target_update_counter % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def update_replay_memory(self, state, action, reward, new_state, done):
        self.replay_memory.append((state, action, reward, new_state, done))

    def get_qs(self, state):
        return self.model(state).cpu().detach().numpy()

    def act_epsilon_greedy(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.output_size)
        else:
            return int(np.argmax(self.get_qs(state)))

    def epsilon_decay(self):
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY
            self.epsilon = max(MIN_EPSILON, self.epsilon)

    def save(self):
        torch.save(self.model.state_dict(), "./qlearning/qpolicy.pth")

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        mini_batch = random.sample(self.replay_memory, self.mini_batch_size)

        states, actions, rewards, new_states, dones = zip(*mini_batch)

        # Convert to numpy arrays first
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        new_states = np.array(new_states)
        dones = np.array(dones)

        # Convert to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        new_states = torch.tensor(new_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Compute Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(new_states).max(1)[0]
        target_q_values = rewards + self.discount_factor * next_q_values * (1 - dones)

        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_target_network()
        self.epsilon_decay()

        return loss.item()

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

    def reset(self):
        self.position = self.start_position
        self.speed = 0
        self.angle = 0
        self.distance = 0

    def action(self, state):

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
        distance_reward = self.distance / (CAR_SIZE_X / 2)
        velocity_reward = self.speed  # Assuming max speed is 20, adjust as needed

        # Combine the rewards (you can adjust the weights)
        total_reward = 0.7 * distance_reward + 0.3 * velocity_reward

        return total_reward
