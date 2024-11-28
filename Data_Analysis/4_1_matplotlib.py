# 4_1_matplotlib.py

import matplotlib.pyplot as plt
import numpy as np
# from matplotlib import cm # colormap
from matplotlib import colormaps as cm
# a = [1, 3, 5, 7]
# b = a # 얅은 복사
# c = a.copy() # 깊은 복사

#
# a[0] = 99
# print(a)
# print(b)
# print(c)

def plot1():
    print(plt.style.available)
    # ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-v0_8',
    # 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted',
    # 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white',
    # 'seaborn-v0_8-whitegrid', 'tableau-colorblind10']
    print(len(plt.style.available)) # 28
    x = np.arange(5)

    # with plt.style.context('dark_background'):
    #     plt.bar(x, x+1)
    #     plt.show()
    # 퀴즈: 가능한 모든 스타일을 1개의 피겨에 그려주세요.
    plt.figure(figsize = (28,28)) # fifure 크기 조정
    for i in range(len(plt.style.available)):
        with plt.style.context(plt.style.available[i]):
            plt.title(plt.style.available[i])
            plt.subplot2grid((6, 5), ((i//5), (i%5)))
            plt.bar(x, x+1)
    plt.tight_layout() # 자동으로 딱 붙게 만든다.
    plt.savefig('style.png') # 저장
    plt.show()


# 퀴즈: x 모양으로 점을 출력하세요.
def plot2():
    # plt.plot(range(10+4), range(10+4),'ro')
    # plt.plot(range(10+4), range(10+4,0,-1),'ro')
    y = np.arange(100)
    plt.scatter(y,y, s = 30, c=y, cmap = 'twilight') # 산점도 전용 함수 s = 점 크기, c =  색상
    plt.scatter(y[::-1], y, s = 20, c=y, cmap = 'plasma') # cmap = 색상 무제한 데이터
    plt.show()


def plot3():
    print(len(plt.colormaps())) # 166
    print(plt.colormaps()) # ['magma', 'inferno', 'plasma', 'viridis', 'cividis', 'twilight', 'twilight_shifted', ...]

def plot_4():
    print('hello world')
    jet = cm['jet']
    print(jet(0))
    print(jet(127)) # R G B A(alpha)
    print(jet(255))
    print(jet(-1)) # 1~255 범위 이외에는 0이전 같은 색깔, 255 이후 같은 색깔
    print(jet(256))

    cols = [jet(0), jet(127), jet(255), jet(-1), jet(256)]


# plot1()
# plot2()
# plot3()
plot_4()
