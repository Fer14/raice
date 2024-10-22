# RAICE 🤖🏎️🏁


(STILL ON DEVELOPMENT)

<p align="center">
  <img src="./logos/big.png" width="700" />
</p>


## INTRODUCTION

RL agents are trained on a custom made racing game. The goal is train multiple algorithms to race on the same tracks and see which one is the best across all F1 tracks.


## RACING CAR-ALGORITHMS


<p align="center">
  <img src="./logos/race_no_bg.png" width="600" />
</p>

- `Blue Cars`: Correspond to Policy-Based Algorithms (e.g., Proximal Policy Optimization, PPO). These agents learn a policy directly by optimizing the expected return, focusing on selecting the best action in each state.

- `Green Cars`: Represent Value-Based Algorithms (e.g., Deep Q-Learning, DQN). These agents learn to estimate the value of actions and states, aiming to improve decision-making based on long-term expected rewards.

- `Both Colors (Blue + Green)`: These cars utilize Hybrid Algorithms that combine both policy-based and value-based approaches, such as Actor-Critic methods, where one part learns the policy and another learns the value function.

- `White Cars`: Indicate the use of a Genetic Algorithm. These cars evolve over time through selection, mutation, and crossover, mimicking natural evolution to optimize their behavior. 


## RAICE EXAMPLE
[Videograbación 2024-09-21 15 07 26.webm](https://github.com/user-attachments/assets/b8581979-502f-4e1b-b7b0-6c8af95dbe04)


## CURRENT MAPS

| TRACK | IMAGE | READY |
|----------|----------|----------|
| 🇧🇭 BAHREIN | ![logo](maps/bahrain2.png) | YES |


## Next Steps

- [x] Train PPO (Proximal Policy Optimization)
- [x] Train A2C (Advantage Actor-Critic)
- [x] Test all cars on the test track
- [x] Create a reinforcement learning course explaining the project and algorithms
- [ ] Add more tracks for evaluation
- [ ] Create a competition using all available tracks
- [ ] Add a track/circuit creator tool
