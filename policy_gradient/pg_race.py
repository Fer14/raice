from car import Car
from race import Race
import sys
import pygame
from pg_car import PGCar

CAR_SIZE_X = 30
CAR_SIZE_Y = 30


class PGRace(Race):

    def __init__(self, start: list[int]) -> None:
        super().__init__(start)
        # calculate the distance between the start and the finish following the black lines
        self.finish = self.start

    # def draw(self, cars: list[Car], draw_radar=False):
    #     self.screen.blit(self.game_map, (0, 0))
    #     for car in cars:
    #         if car.is_alive():
    #             car.draw(self.screen, draw_radar)

    #     for point in self.points:
    #         pygame.draw.circle(self.screen, (255, 0, 0), point, 5)

    #     # draw line between points
    #     for i in range(len(self.points) - 1):
    #         pygame.draw.line(
    #             self.screen,
    #             (255, 0, 0),
    #             self.points[i],
    #             self.points[i + 1],
    #             2,
    #         )

    #     pygame.display.flip()

    def step(self, car):
        new_state = car.get_data()
        reward = car.get_reward()
        done = not car.is_alive()
        return new_state, reward, done

    def training_race(self, car: PGCar, episodes, train_every):

        clock = pygame.time.Clock()

        for episode in range(1, episodes + 1):

            current_state = car.get_data()
            states, actions, rewards = [], [], []

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
                states.append(current_state)
                actions.append(action)
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
