import matplotlib.pyplot as plt
import numpy as np

# x축 y축
def plot_1():
    y = [1, 2, 3, 4, 5]

    plt.plot(y)
    plt.show()

def plot_2():
    x = [1, 2, 3, 4, 5]
    y = [1, 2, 3, 4, 5]


    plt.plot(x, y) # Line
    plt.plot(x, y,'ro') # scatter
    # 퀴즈: 검정색 x 산점도를 그려보세요
    plt.plot(x, y,'kx') # scatter
    plt.show()

# 퀴즈: y = x^2 그래프를 그리는 함수를 만드세요.

def squre_1():
    # x = np.arange(-10,11, 1)
    x = np.linspace(-10,10,100)
    y = np.square(x)
    plt.plot(x,y,'ro')
    plt.plot(x,y)
    plt.show()

def plot_3():
    x = np.arange(10)

    plt.subplot(2, 2, 1)
    plt.title('x*2')
    plt.plot(x, x*2, 'r')
    plt.subplot(2, 2, 2)
    plt.title('x*3')
    plt.plot(x, x*3, 'g')
    plt.subplot(2, 2, 3)
    plt.title('x**2')
    plt.plot(x, x**2, 'b')
    plt.subplot(2, 2, 4)
    plt.title('x**3')
    plt.plot(x, x**3, 'y')

    plt.show()

# desmos 에서 그렸던 로그 그래프를 피켜 1개에 서브 플랏 4개를 사용해서 그려보세요.
def plot_4():
    x = np.linspace(-10,10,100)

    plt.subplot(2, 2, 1)
    plt.title('log(x)')
    plt.plot(x, np.log(x), 'r')
    plt.subplot(2, 2, 2)
    plt.title('-log(x)')
    plt.plot(x, -np.log(x), 'g')
    plt.subplot(2, 2, 3)
    plt.title('log(-x)')
    plt.plot(x, np.log(-x), 'b')
    plt.subplot(2, 2, 4)
    plt.title('-log(-x)')
    plt.plot(x, -np.log(-x), 'y')

    plt.show()

# 전체를 채우는 그림을 그리세요.
def plot_5():
    plt.subplot2grid((4, 4),(0, 2), rowspan=3)
    plt.plot(range(5), range(5))
    plt.subplot2grid((4, 4), (3, 2), rowspan=4)
    plt.plot(range(5), range(5))
    plt.subplot2grid((4, 4), (0, 0), rowspan=4)
    plt.plot(range(5), range(5))
    plt.subplot2grid((4, 4), (0, 1), rowspan=4)
    plt.plot(range(5), range(5))
    plt.subplot2grid((4, 4), (0, 3), rowspan=4)
    plt.plot(range(5), range(5))
    plt.show()

def plot_6():
    boys = [30, 21, 24, 28, 25]
    girls = [33, 37, 19, 22, 31]
    x = range(len(boys))
    y = np.arange(len(girls))
    plt.bar(x, boys, width= 0.4)
    plt.bar(y+0.5, girls, width= 0.4)
    plt.xticks(y+0.25,['A', 'B', 'C', 'D', 'E'])
    plt.show()


# 퀴즈
# plot_1()
# plot_2()
# squre_1()
# plot_4()
# plot_5()
plot_6()


