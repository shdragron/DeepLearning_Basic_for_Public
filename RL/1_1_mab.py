# 1_1_mab.py
# mab: Multi-Armed Bandits
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)


# exploration, exploitation
def greedy(means):
    return np.argmax(means)         # exploitation


def e_greedy(means, e): # e: 입실론 -> 0.1dl
    if np.random.rand() < e:
        return np.random.choice(range(len(means))) # exploration

    return np.argmax(means)


# 퀴즈
# rand, randn, choice 함수를 1만번 호출한 결과를 그래프로 그려보세요(히스토그램)
def show_randoms():
    n0, n1, n2 = [], [], []
    for _ in range(10000):
        n0.append(np.random.rand())
        n1.append(np.random.randn())        # n: normal
        n2.append(np.random.choice(30))

    plt.subplot(1, 3, 1)
    plt.hist(n0)

    plt.subplot(1, 3, 2)
    plt.hist(n1)

    plt.subplot(1, 3, 3)
    plt.hist(n2)

    plt.show()

def show_casino(bandits, N):
    means = [0] * len(bandits)
    samples = [0] * len(bandits)

    for _ in range(N):
        select = greedy(means) # means:  [0.989 0.    0.   ]
        # select = e_greedy(means, e = 0.1) # means:  [ 1.014 -2.011  3.009]

        # print(select)

        reward = bandits[select] + np.random.randn() # -4.5 ~ 4.5

        samples[select] += 1
        ratio = 1 / samples[select]
        means[select] = (1 - ratio) * means[select] + ratio * reward
        # print(means)
    print('means: ' , np.array(means))
    # print('argmax: ', np.argmax(means))
    # print('samples: ', samples)

bandits = [1.0, -2.0, 3.0]
show_casino(bandits, 10000)

# show_randoms()








