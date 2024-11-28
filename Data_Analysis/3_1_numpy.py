# 3_1_numpy.py
import numpy as np

print(np.zeros([3, 4],dtype=np.int32))

print(np.ones((3, 4), dtype = np.float32))

print(np.full((3, 4), 5))


a = np.zeros([4, 5],dtype=np.uint8)
print(a)

# 퀴즈: 테두리를 1로 체우세요.
a[0,:], a[3,:], a[:,0], a[:,4] = 1, 1, 1, 1

print(a)

 # 퀴즈: 속을 2로 채우세요.

a[1,1:-1], a[2,1:-1] = 2, 2
print(a)

# 퀴즈: 대각선 방향으로 3으로 채우세요.
b = np.zeros([5, 5],dtype=np.uint8)

b[[0,1,2,3,4],[0,1,2,3,4]] # index 배열
b[range(5),list(reversed(range(5)))] # index 배열

print(b)
print()

c = np.array([11, 22, 33, 44, 55])

c[[0,1]] = 77 # index array

print(c)

# 배열을 확 바꿀 때 사용.
d = np.arange(len(c))
np.random.shuffle(d)

print(c[d])
print(d)
print()
# 퀴즈: 2차원 배열을 거꾸로 출력하세요.

e = np.random.randint(0,100,12).reshape(-1,4)

print(e)

print(e[::-1,::-1]) # 거꾸로 뒤집는 코드

print('-'*40)


f = np.arange(6).reshape(2,3)
print(f)
print(f + f) # 백터 연산으로 2행 3열이 나온다.
print(np.concatenate([f,f])) # 4행 3열이 되는 문법
print(np.concatenate([f,f],axis=1)) # 2행 6열이 된다. axis = 1이 수평
print('-'*40)
# 2차원 한정 동작
print(np.hstack([f,f]))
print(np.vstack([f,f]))
print('-'*40)

# 퀴즈: 2차원 배열을 행과 열을 바꿔서 출력하세요.
for i in range(3):
    print(f[:,i])
# transpose 연산
print(np.transpose(f))
print(f.T)

# 행렬 곱셈(점곱 연산, point-wise)
# (2, 3) @ (2, 3)
# 동작을 안한다. @ 행렬곱 연산이기 때문에.
# (2, 3) @ (3, 2) -> Transpose = (2, 2)
# 동작을 안한다. @ 행렬곱 연산이기 때문에.
# (3, 2) @ (2, 3) = (3, 3)

#3,4 @ 4,3 = (3, 3)
print(np.dot(e,e.T))





