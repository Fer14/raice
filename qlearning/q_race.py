from car import Car
from race import Race
import sys
import pygame
from q_car import QCar

CAR_SIZE_X = 30
CAR_SIZE_Y = 30


class QRace(Race):

    def __init__(self, start: list[int]) -> None:
        super().__init__(start)
        # calculate the distance between the start and the finish following the black lines
        self.finish = self.start

    def step(self, car):
        new_state = car.get_data()
        reward = car.get_reward()
        done = not car.is_alive()
        return new_state, reward, done

    def training_race(self, car: QCar, episodes, train_every):

        clock = pygame.time.Clock()

        for episode in range(1, episodes + 1):

            current_state = car.get_data()

            done = False
            episode_reward = 0
            while not done:

                clock.tick(500)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit()
                        sys.exit()

                action = car.action(current_state)
                car.update(self.game_map)
                new_state, reward, done = self.step(car)
                episode_reward += reward

                current_state = new_state
                self.draw([car], draw_radar=True)
                car.update_replay_memory(current_state, action, reward, new_state, done)

            if episode % train_every == 0:
                loss = car.train()
            if episode % 100 == 0:
                print(
                    f"Episode {episode}, Mean Episode Reward: {episode_reward/train_every:.2f}, Loss: {loss:.4f} "
                )
