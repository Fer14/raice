from race import Race
import sys
import pygame
from ppo_car import PPOCar
import torch

CAR_SIZE_X = 30
CAR_SIZE_Y = 30


class PPORace(Race):

    def __init__(self, start: list[int], finnish_line: tuple[int]) -> None:
        super().__init__(start, finnish_line)
        # calculate the distance between the start and the finish following the black lines

    def step(self, car):
        new_state = car.get_data()
        reward = car.get_reward()
        done = not car.is_alive()
        return new_state, reward, done

    def training_race(self, car: PPOCar, episodes, train_every):

        clock = pygame.time.Clock()
        ppo_epochs = 4

        for episode in range(1, episodes + 1):

            current_state = car.get_data()
            rewards = []

            done = False
            episode_reward = 0
            while not done:

                clock.tick(500)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit()
                        sys.exit()

                action = car.action_train(current_state)
                car.update(self.game_map)

                new_state, reward, done = self.step(car)

                rewards.append(reward)
                episode_reward += reward

                current_state = new_state
                self.draw([car], draw_radar=True)

            if episode % train_every == 0:
                loss = car.train(rewards)
            if episode % 100 == 0:
                print(
                    f"Episode {episode}, Mean Episode Reward: {episode_reward/train_every:.2f}, Loss: {loss:.4f} "
                )
