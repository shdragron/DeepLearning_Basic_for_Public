# 2_4_q_learning.py
import gym
import numpy as np
import util


def simulation():
    env = gym.make('FrozenLake-v1', render_mode='human', is_slippery=False)

    q_table = np.zeros((16, 4))

    for i in range(6):
        state, _ = env.reset()
        # print(state)

        for action in [2, 2, 1, 1, 1, 2]:
            next_state, reward, done, _, _ = env.step(action)

            # q_table[state, action] = 현재 보상 + 다음 상태에서의 누적 보상
            q_table[state, action] = reward + np.max(q_table[next_state])
            state = next_state

        # print(q_table)
        util.draw_q_table(q_table)


# 현재 상태에서의 최대 보상에 대한 인덱스(LEFT, DOWN, RIGHT,UP) 반환
# 최대값이 여러 개 있을 때, 랜덤하게 선택
def random_argmax(rewards):
    r_max = np.max(rewards)

    # print(rewards[rewards == r_max])          # 가장 큰 값 출력
    indices = np.nonzero(rewards == r_max)      # 가장 큰 값 인덱스
    # print(indices, type(indices))
    # print(indices[0])

    return np.random.choice(indices[0])


# random_argmax(np.array([2, 3, 6, 4, 6]))

# 퀴즈
# random_argmax 함수를 완성하고
# 2000번 반복했을 때, 성공 횟수를 알려주세요
def q_learning():
    env = gym.make('FrozenLake-v1', is_slippery=False)  # , render_mode='human'
    q_table = np.zeros((16, 4))

    success = 0
    for i in range(2000):
        state, _ = env.reset()

        done = False
        while not done:
            action = random_argmax(q_table[state])
            next_state, reward, done, _, _ = env.step(action)

            q_table[state, action] = reward + np.max(q_table[next_state])
            state = next_state

        success += reward
        if i % 10 == 0:
            print(i)

    util.draw_q_table(q_table)
    print('성공 :', success, success/2000)


# simulation()
q_learning()
