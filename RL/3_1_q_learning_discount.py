# 3_1_q_learning_discount.py
import gym
import numpy as np
import util
import matplotlib.pyplot as plt


def random_argmax(rewards):
    r_max = np.max(rewards)
    indices = np.nonzero(rewards == r_max)

    return np.random.choice(indices[0])


def e_greedy(i, env, actions):
    # e = 1 / (i // 100 + 1)
    e = 0.1 / (i + 1)
    # return np.random.randint(4) if np.random.rand() < e else random_argmax(actions)  # np.argmax(actions)

    if np.random.rand() < e:
        return env.action_space.sample()

    return random_argmax(actions)


def random_noise(i, env, actions):
    # values = actions + np.random.rand(len(actions))
    # values = actions + np.random.randn(len(actions))        # 잘 안나옴. -4.5 ~ 4.5
    values = actions + np.random.randn(env.action_space.n) / (i + 1)
    return np.argmax(values)


def q_learning_discount():
    env = gym.make('FrozenLake-v1', is_slippery=False)
    q_table = np.zeros((16, 4))

    success, discounted = 0, 0.9
    results = []
    for i in range(2000):
        state, _ = env.reset()

        done = False
        while not done:
            # action = e_greedy(i, env, actions=q_table[state])
            action = random_noise(i, env, actions=q_table[state])
            next_state, reward, done, _, _ = env.step(action)

            q_table[state, action] = reward + discounted * np.max(q_table[next_state])
            state = next_state

        success += reward
        results.append(reward)
        if i % 10 == 0:
            print(i)

    util.draw_q_table(q_table)
    print('성공 :', success, success/2000)

    plt.plot(range(len(results)), results)
    plt.tight_layout()
    plt.show()


q_learning_discount()
