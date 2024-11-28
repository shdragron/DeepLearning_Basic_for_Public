# 3_3_q_network.py

import gym
import keras
import numpy as np
from pyparsing import actions

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


# 숫자 1개 -> (1, 16)

def make_onehot(state): # state -> 16개 칸
    z = np.zeros(16)
    z[state] = 1
    return z.reshape(1, -1)

def q_network(loop): # state input , action output
    env = gym.make('FrozenLake-v1', is_slippery=False)   # stochastic world
    # q_table = np.zeros((16, 4))

    model = keras.models.Sequential([

        keras.layers.Input(shape = (16, )),
        keras.layers.Dense(4, use_bias=False)  # action 4개
    ])

    model.compile(loss='mse', optimizer=keras.optimizers.SGD(0.1))

    success, discounted = 0, 0.9
    results = []
    for i in range(loop):
        state, _ = env.reset()

        done = False
        while not done:
            p = model.predict(make_onehot(state), verbose=0)
            actions = p[0]

            # action = e_greedy(i, env, actions=q_table[state])
            action = random_noise(i, env, actions)
            next_state, reward, done, _, _ = env.step(action)

            if done:
                p[0, action] = reward
            else:
                p_next = model.predict(make_onehot(next_state), verbose=0)
                p[0, action] = reward + discounted * np.max(p_next[0])

            model.fit(make_onehot(state), p, verbose=1, epochs=1)
            state = next_state

        success += reward
        results.append(reward)
        # if i % 10 == 0:
        #     print(i)

    # util.draw_q_table(q_table)
    print('성공 :', success, success/loop)

    # plt.plot(range(len(results)), results)
    # plt.tight_layout()
    # plt.show()


q_network(2000)

# black/white
# * * *
