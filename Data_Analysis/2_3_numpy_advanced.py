# 2_3_numpy_advanced.py
import numpy as np
np.random.seed(41)
# 원래는 균등 -> 실제는 정규분포표를 따흔다.

a = np.random.randn(3, 4)
print(a)

b =np.random.randint(0, 100, 12).reshape(3, 4)
print(b)

# 퀴즈: 2차원 넘파이 배열에서 가장 큰 값을 찾으세요.
tmp = 0
for i in b:
    for j in i:
        if j > tmp:
            tmp = j
        else:
            pass

print(tmp)
print(max([max(i) for i in b]))
print()

print(b.max(axis=0)) # 수직 52,81,13 중 큰거 81
print(b.max(axis=1)) # 수평

# 중요하다
print(np.argmax(b))
print(np.argmax(b, axis=0))
print(np.argmax(b, axis=1))
print()


# 퀴즈: 넘파이 2차원 배열에서 첫 번째와 마지막 번째의 요소를 출력하세요.

print(b[0][0],b[2][3])
print(b[0, 0],b[2, 3]) # fancy indexing
print()

# 퀴즈: 넘파이 2차원 배열의 마지막 행을 출력하고 마지막 열을 출력하세요.
for n in range(len(b)):
    print(b[n, 3])
print(b[:,3])