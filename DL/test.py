# import pandas as pd
# import numpy as np
# from sklearn import preprocessing
# import keras
# import matplotlib.pyplot as plt
#
# data = pd.read_csv('data/trees.csv')
#
# print(data.columns)
#
# x_1 = data.rownames.values.reshape(-1,1)
# x_2 = data.Girth.values.reshape(-1,1)
# x_3 = data.Height.values.reshape(-1,1)
# y = data.Volume.values.reshape(-1,1)
#
# x = np.hstack((x_1, x_2, x_3))
#
# # x = preprocessing.minmax_scale(x)
# # loss 15.7
# # 그냥
# # 25.3
# # x = preprocessing.scale(x)
# # 10.4
#
# model = keras.models.Sequential([keras.layers.Dense(units=1)])
# # model.add(keras.layers.Dense(units=1))
# # 모델 컴파일
# model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.5),
#               loss=keras.losses.MeanSquaredError,
#               metrics = ['RootMeanSquaredError']
#               )
# # 모델 훈련
# model.fit(x, y, epochs = 1000,verbose = 1) # 순전파 + 최적화함수 + 역전파
import numpy as np
from sklearn import preprocessing

enc = preprocessing.OneHotEncoder()

a = np.array([40,20,10]).reshape(-1,1)

a= arg

onehot = np.eye(3)
print(onehot)

print(onehot.shape)
print('-'*100)
print(onehot[a])
print(enc.fit_transform(a))
b = enc.fit_transform(a)
b = np.array(b)
print(onehot[b])