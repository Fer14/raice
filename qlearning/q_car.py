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

REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 1000
MIN_EPSILON = 0.1
EPSILON_DECAY = 0.9999


class QCar(Car):

    def __init__(
        self,
        position=None,
        device="cuda",
        input_size=5,
        hidden_size=5,
        output_size=4,
        discount_factor=0.99,
        learning_rate=1e-3,
        mini_batch_size=256,
        update_target_every=50,  # 150
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
        # self.replay_buffer = PrioritizedReplayBuffer(REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

        self.discount_factor = discount_factor
        self.epsilon = 1
        self.mini_batch_size = mini_batch_size
        self.output_size = output_size
        self.update_target_every = update_target_every

        self.sprite = pygame.image.load("./qlearning/car.png").convert_alpha()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        self.crashed = False
        self.last_position = self.position

    def create_model(self, input_size, hidden_size, output_size, trainable=True):
        model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
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
        # self.replay_buffer.push(state, action, reward, new_state, done)

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
        torch.save(self.model.state_dict(), "./qlearning/qpolicy.pth")

    def load(self):
        self.model.load_state_dict(torch.load("./qlearning/bestqpolicy.pth"))

    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        mini_batch = random.sample(self.replay_memory, self.mini_batch_size)
        # mini_batch, indices, weights = self.replay_buffer.sample(self.mini_batch_size)

        states, actions, rewards, new_states, dones = zip(*mini_batch)

        # Convert to PyTorch tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        new_states = torch.tensor(np.array(new_states), dtype=torch.float32).to(
            self.device
        )
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device)
        # weights = torch.tensor(np.array(weights), dtype=torch.float32).to(self.device)

        # Compute Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Use the online network to select actions for the next state
        next_actions = self.model(new_states).argmax(1).unsqueeze(1)

        next_q_values = self.target_model(new_states).gather(1, next_actions).squeeze(1)
        target_q_values = rewards + self.discount_factor * next_q_values * (1 - dones)

        loss = nn.MSELoss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # td_errors = (target_q_values - current_q_values).detach().cpu().numpy()
        # self.replay_buffer.update_priorities(indices, td_errors)

        self.update_target_network()
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
            self.n_drifts_left += 1
            self.n_drifts_right = 0
        elif action == 1:
            self.angle -= 10  # Right
            self.n_drifts_left = 0
            self.n_drifts_right += 1
        elif action == 2:
            if self.speed - 2 >= 6:
                self.speed -= 2  # Slow Down
            self.n_drifts_right = 0
            self.n_drifts_left = 0
        else:
            self.n_drifts_right = 0
            self.n_drifts_left = 0
            if self.speed + 2 <= 12:
                self.speed += 2  # Speed Up

        return action

    def get_reward(self):
        if self.crashed:
            self.crashed = False
            return -100

        # Calculate reward based on distance and velocity
        distance_traveled = np.linalg.norm(
            np.array(self.position) - np.array(self.last_position)
        )
        distance_reward = self.distance / (CAR_SIZE_X / 2)
        velocity_reward = self.speed

        total_reward = (
            0.1 * distance_reward + 0.1 * velocity_reward + 0.8 * distance_traveled
        )

        total_reward = 0.7 * distance_reward + 0.3 * velocity_reward

        self.last_position = self.position

        return total_reward
