# 2_2_monte_carlo.py
# 퀴즈
# 몬테카를로 방법론을 이용해서 원주율을 계산하는 코드로 만드세요
import numpy as np
from matplotlib import pyplot as plt

N = 10000
cnt = 0
inner, outer = [],[]
for i in range(N):
    x = np.random.rand()
    y = np.random.rand()

    cnt += (x**2 + y**2 < 1)

    if x**2 + y**2 < 1:
        inner.append((x,y))
    else:
        outer.append((x,y))

print(cnt*4/N)

for x, y in inner:
    plt.plot(x, y,'ro')
for x, y in outer:
    plt.plot(x, y, 'bo')
plt.show()