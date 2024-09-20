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


class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, output_size), nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        shared = self.shared(x)
        return self.actor(shared), self.critic(shared)


class A2Car(Car):

    def __init__(
        self,
        position=None,
        device="cuda",
        input_size=5,
        hidden_size=5,
        output_size=4,
        discount_factor=0.99,
        learning_rate=1e-3,  # best is 1e-3
        entropy_weight=0.01,
    ):
        super().__init__(position=position, angle=0)
        self.device = device
        self.model = ActorCritic(input_size, hidden_size, output_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.discount_factor = discount_factor
        self.entropy_weight = entropy_weight

        self.sprite = pygame.image.load("./a2c/car.png").convert_alpha()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        self.crashed = False
        self.name = "A2C"

    def save_policy(self):
        torch.save(self.model.state_dict(), "./a2c/policy.pth")

    def load_policy(self):
        self.model.load_state_dict(torch.load("./a2c/policy_best.pth"))

    def reset_episode(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs, value = self.model(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        entropy = m.entropy()

        self.log_probs.append(log_prob)
        self.values.append(value)
        self.entropies.append(entropy)

        return action.item()

    def train(self):
        returns = []
        R = 0
        for reward in self.rewards[::-1]:
            R = reward + self.discount_factor * R
            returns.insert(0, R)

        returns = torch.tensor(returns).to(self.device)
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values).squeeze()
        entropies = torch.stack(self.entropies)

        advantages = returns - values.detach()

        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        entropy_loss = -entropies.mean()

        loss = actor_loss + 0.5 * critic_loss + self.entropy_weight * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.reset_episode()

        return loss.item()

    def action_train(self, state):

        action = self.select_action(state)

        if action == 0:
            self.angle += 10  # Left
        elif action == 1:
            self.angle -= 10  # Right
        elif action == 2:
            if self.speed - 2 >= 6:
                self.speed -= 2  # Slow Down
        else:
            self.speed += 2  # Speed Up

    def action(self):
        state = self.get_data()
        action = self.select_action(state)

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
