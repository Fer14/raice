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


class PPOCar(Car):

    def __init__(
        self,
        position=None,
        device="cuda",
        input_size=5,
        hidden_size=64,
        output_size=4,
        discount_factor=0.99,
        learning_rate=1e-4,
        epsilon=0.2,
        entropy_weight=0.01,
        critic_weight=0.5,
        ppo_epochs=5,
        batch_size=64,
    ):
        super().__init__(position=position, angle=0)
        self.device = device
        self.model = self.model = ActorCritic(input_size, hidden_size, output_size).to(
            self.device
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.entropy_weight = entropy_weight
        self.critic_weight = critic_weight

        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        self.sprite = pygame.image.load("./ppo/car.png").convert_alpha()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        self.reset_episode()
        self.crashed = False
        self.name = "PPO"

    def save_policy(self):
        torch.save(self.model.state_dict(), "./ppo/policy.pth")

    def load_policy(self):
        self.model.load_state_dict(torch.load("./ppo/best_policy.pth"))

    def reset_episode(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.entropies = []

    def forward(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs, value = self.model(state)
        m = Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        entropy = m.entropy()

        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.entropies.append(entropy)

        return action.item()

    def compute_gae(self, next_value, rewards):
        gae = 0
        returns = []
        values = self.values + [next_value]
        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step] + self.discount_factor * values[step + 1] - values[step]
            )
            gae = delta + self.discount_factor * 0.95 * gae
            returns.insert(0, gae + values[step])
        return returns

    def train(self, rewards):
        next_state = torch.FloatTensor(self.get_data()).unsqueeze(0).to(self.device)
        _, next_value = self.model(next_state)
        returns = self.compute_gae(next_value, rewards)

        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        old_log_probs = torch.cat(self.log_probs).detach()
        returns = torch.tensor(returns).to(self.device)

        advantages = returns - torch.cat(self.values).detach().squeeze()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.ppo_epochs):
            # Create mini-batches
            indices = torch.randperm(len(states))
            for start in range(0, len(states), self.batch_size):
                end = min(start + self.batch_size, len(states))
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                probs, values = self.model(batch_states)
                m = Categorical(probs)
                new_log_probs = m.log_prob(batch_actions)
                entropy = m.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
                    * batch_advantages
                )

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (batch_returns - values.squeeze()).pow(2).mean()
                entropy_loss = -entropy

                loss = (
                    actor_loss
                    + self.critic_weight * critic_loss
                    + self.entropy_weight * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

        self.reset_episode()
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
