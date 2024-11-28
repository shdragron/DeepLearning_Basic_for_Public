# 1_1_linear_regression_python.py
import matplotlib.pyplot as plt
import cmath
# y = ax + b -> 정답

# H(x) = Wx + b -> 가설 예측 값

# 목표: H(x) 값을 y랑 가깝게 만드는 것

# 예측값 : inference

# 입력은 그대로, w가 바뀐다. 여러 epoch 기준

# 손실함수 MSE: Mean Squre Error

def cost(x, y, w): # max: -3.0: 5.333
    sum = 0
    for i in range(len(x)):
        hx = w * x[i]   # prediction
        sum += (hx - y[i]) ** 2
        return sum / len(x)

# 손실함수 MAE: Mean Absolute Error

# def cost(x, y, w): # max: -3.0: 1.333
#     sum = 0
#     for i in range(len(x)):
#         hx = w * x[i]   # prediction
#         sum += abs(hx - y[i])
#         return sum / len(x)

# 손실함수 RMSE: Root Mean Squre Error

# def cost(x, y, w): # max: -3.0: 1.333
#     sum = 0
#     for i in range(len(x)):
#         hx = w * x[i]   # prediction
#         sum += (hx - y[i]) ** 2
#         return (sum / len(x))**(1/2)
#
# # 우리의 목표
# # y  = ax + b
# #      1    0
# # hx = wx + b
#

# 경사하강법은 손실함수의 미분이다.
def gradient_descent(x, y, w):
    loss = 0
    for i in range(len(x)):
        hx = w * x[i]           # prediction, inference
        # loss += (hx - y[i]) ** 2 미분
        # loss += 2 * (w * x[i] - y[i] **  (2-1) * (w * x[i] - y[i]))<- 항만 미분
        # loss += 2 * (w * x[i] - y[i] **  (2-1) * x[i])
        # loss += 2 * (hx - y[i] **  (2-1) * x[i])
        # loss  += 2 * ((hx - y[i]) * x[i])
        loss += (hx - y[i]) * x[i]

    return loss / len(x)

def show_cost():
    x = [1, 2, 3]
    y = [1, 2, 3]
    for i in range(-30,50):
        w = i/10
        loss = cost(x, y, w)
        # print(w, loss)
        plt.plot(w, loss, 'ro')
    plt.show()

def pos_show_gradient():
    x = [1, 2, 3]
    y = [1, 2, 3]
    w = 5
    while(True):
        if w == 1.0: # 
            print(w)
            break
        else:
            g = gradient_descent(x, y, w)
            w -= 0.0001 * g
            print(w)

def nev_show_gradient():
    x = [1, 2, 3]
    y = [1, 2, 3]
    w = -5
    for i in range(5):
        g = gradient_descent(x,y,w)
        w -= 0.1 * g # 0.1은 learning rate
        print(w)

# show_cost()
pos_show_gradient()
print()
# nev_show_gradient()

# 미분: 기울기
#      x가 1만큼 이동했을때 y가 이동한거리

# y = x
# w가 loss에 미치는 영향이 기울기다.


