# 1_1_linear_regression_python.py
import matplotlib.pyplot as plt


def cost(x, y, w):
    c = 0
    for i in range(len(x)):
        hx = w * x[i]           # prediction, inference
        c += (hx - y[i]) ** 2

    return c / len(x)


def gradient_descent(x, y, w):
    c = 0
    for i in range(len(x)):
        hx = w * x[i]
        c += (hx - y[i]) * x[i]
        # c += (hx - y[i]) ** 2
        # c += (w * x[i] - y[i]) ** 2
        # c += 2 * (w * x[i] - y[i]) ** (2-1) * (w * x[i] - y[i])미분
        # c += 2 * (w * x[i] - y[i]) * x[i]
        # c += (hx - y[i]) * x[i]

    return c / len(x)


def show_cost():
    # y  = ax + b
    #      1    0
    # hx = wx + b
    x = [1, 2, 3]
    y = [1, 2, 3]

    # 퀴즈
    # w와 c를 그래프에 표시하세요
    for i in range(-30, 50):
        w = i / 10
        c = cost(x, y, w)
        # print(w, c)

        plt.plot(w, c, 'ro')
    plt.show()


def show_gradient():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = 5
    for i in range(5):
        c = cost(x, y, w)
        g = gradient_descent(x, y, w)
        w -= 0.1 * g

        print(c)

    # 퀴즈
    # w를 1.0으로 만드세요 (3가지)


# show_cost()
show_gradient()

# 미분: 기울기, 순간변화량
#       x가 1만큼 움직였을 때 y가 움직인 거리

# y = 7         7=1, 7=2, 7=3
# y = x         1=1, 2=2, 3=3
# y = (x + 1)   2=1, 3=2, 4=3
# y = 2x        2=1, 4=2, 6=3
# y = xz

# y = x ^ 2         1=1, 4=2, 9=3
#     x ^ 2 = 2 * x ^ (2 - 1) * x미분 = 2x
# y = (x + 1) ^ 2
#     (x + 1) ^ 2 = 2 * (x + 1) ^ (2 - 1) * (x + 1)미분 = 2(x+1)
