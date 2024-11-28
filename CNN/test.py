import numpy as np

a = [
    [1,2,3],
    [3,2,1]
]

b = np.array(a)

c = np.argmax(b,axis=0)
d = np.argmax(b,axis=1)
print(b.shape)
print(b)
print(c)
print(d)

d = [1,2,4]
d = np.array(d)
print(len(d))
d = [0] * d.shape[0]

print(d)