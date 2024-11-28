# 4_1_cartpole.py
import gym
import numpy as np

env = gym.make('CartPole-v1', render_mode='human')

print(env.observation_space)
print(env.action_space)

print(env.action_space.n)

state, _ = env.reset()
print(state)
# Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity
# [ 0.01736585 -0.02656827  0.01767714  0.00382694]

# 퀴즈
# Pole Angle만 사용해서 막대를 넘어뜨리지 말아보세요
# print(env.step(0))
# array([-0.02325383, -0.21918887,  0.01235517,  0.32603997], dtype=float32),
# 1.0, False, False, {}

done, stand = False, 0
while not done:
    angle = state[2]
    action = 0 if angle < 0 else 1

    state, reward, done, _, _ = env.step(action)
    stand += 1

env.close()
print(stand)


