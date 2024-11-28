# 3_4_q_network_tf.py
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import util

tf.disable_v2_behavior()


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
def make_onehot(state):
    z = np.zeros(16)
    z[state] = 1
    return z.reshape(1, -1)


def q_network_tf(loop):
    env = gym.make('FrozenLake-v1', is_slippery=False)

    x = tf.placeholder(tf.float32, shape=(1, 16))   # state
    y = tf.placeholder(tf.float32, shape=(1, 4))    # action

    w = tf.Variable(tf.random_uniform([16, 4]))
    # b = tf.Variable(tf.zeros([4]))

    # (1, 4) = (1, 16) @ (16, 4)
    hx = tf.matmul(x, w)  # + b                     # hx = wx + b

    loss = tf.reduce_mean((hx - y) ** 2)            # mean squared error
    train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    # train = tf.train.GradientDescentOptimizer(0.1).minimize(tf.reduce_mean((tf.matmul(x, w) - y) ** 2))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    onehot = np.zeros(16).reshape(1, -1)

    success, discounted = 0, 0.99
    results, episodes = [], []
    for i in range(loop):
        state, _ = env.reset()

        done, cnt = False, 0
        while not done:
            # p = model.predict(make_onehot(state), verbose=0)
            # p = sess.run(hx, {x: make_onehot(state)})       # 결과 (1, 4)
            onehot[0, state] = 1
            p = sess.run(hx, {x: onehot})       # 결과 (1, 4)
            onehot[0, state] = 0

            action = e_greedy(i, env, actions=p[0])
            # action = random_noise(i, env, actions=p[0])
            next_state, reward, done, _, _ = env.step(action)

            if done:
                p[0, action] = reward
            else:
                # p_next = model.predict(make_onehot(next_state), verbose=0)
                # p_next = sess.run(hx, {x: make_onehot(next_state)})
                onehot[0, next_state] = 1
                p_next = sess.run(hx, {x: onehot})  # 결과 (1, 4)
                onehot[0, next_state] = 0
                p[0, action] = reward + discounted * np.max(p_next[0])

            # model.fit(make_onehot(state), p, epochs=1, verbose=0)
            # sess.run(train, {x: make_onehot(state), y: p})
            onehot[0, state] = 1
            sess.run(train, {x: onehot, y: p})
            onehot[0, state] = 0
            state = next_state

            cnt += 1

        success += reward
        results.append(reward)
        episodes.append(cnt)
        if i % 10 == 0:
            print(i)

    # ----------------------------- #
    # state, _ = env.reset()
    # done = False
    # while not done:
    #     p = sess.run(hx, {x: make_onehot(state)})
    #     action = np.argmax(p[0])
    #     state, _, done, _, _ = env.step(action)
    #     print(state, end=' ')
    # print()

    q_table = sess.run(w)
    # print(q_table)

    state, _ = env.reset()
    while state != 15:

        action = np.argmax(q_table[state])
        state, _, done, _, _ = env.step(action)
        print(state, end=' ')
    print()

    # for state in range(16):
    #     p = sess.run(hx, {x: make_onehot(state)})
    #     print(p[0])

    # (1, 4) 16개 -> (16, 4)
    # q_table = np.vstack([sess.run(hx, {x: make_onehot(state)}) for state in range(16)])
    # print(q_table)
    # util.draw_q_table(q_table)

    sess.close()

    # util.draw_q_table(q_table)
    print('성공 :', success, success/loop)

    # 퀴즈
    # 학습한 결과를 토대로 최적의 경로를 출력하세요

    # plt.subplot(1, 2, 1)
    # plt.plot(range(len(results)), results)
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(range(len(episodes)), episodes)
    #
    # plt.tight_layout()
    # plt.show()


q_network_tf(2000)
