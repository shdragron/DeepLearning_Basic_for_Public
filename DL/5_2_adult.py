# 5_2_adult.py

# >50K, <=50K.
#
# age: continuous.
# workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# fnlwgt: continuous.
# education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# education-num: continuous.
# marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# sex: Female, Male.
# capital-gain: continuous.
# capital-loss: continuous.
# hours-per-week: continuous.
# native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

# 퀴즈: 파일에 대해서 80프로 학습 20 정확도를 구하세요.
import pandas as pd
import numpy as np
import keras
from sklearn import preprocessing, model_selection

def read_adult():
    adult = pd.read_csv('data/adult.data', header=None ,delimiter=',')
    print(adult)
    print(adult.values)
    print(adult.describe()) # 데이터 분석 요약
    print(adult.info()) # 데이터 값 형태 및 타입 출력

    enc = preprocessing.LabelEncoder()

    # 1번
    binds = []
    for index in [1,3,5,6,7,8,9,13]:
        binds.append(enc.fit_transform(adult.values[:,index]))
    for index in [0,2,4,10,11,12]:
        binds.append((adult.values[:,index]))
    print(np.array(binds).shape)

    binds = np.array(binds).T
    x = binds
    y = enc.fit_transform(adult.values[:,-1])

    return x,y

#
def read_adult_onehot():
    adult = pd.read_csv('data/adult.data', header=None ,delimiter=',')

    enc = preprocessing.LabelBinarizer()

    # 7 -> label encoder
    # 0 0 0 0 0 0 1 0 0 -> label binarizer
    # 1번
    binds = []
    character = []
    number = []
    for index in [1,3,5,6,7,8,9]:
        character.append(enc.fit_transform(adult.values[:,index]))
    for index in [0,2,4,10,11,12]:
        number.append((adult.values[:,index]))

    binds = []
    numbers = np.array(number).T
    characters = np.hstack(character)
    print(characters.shape)
    print(numbers.shape)
    binds = np.concatenate((characters,numbers),axis=1)
    print(binds)


    x = binds
    y = np.hstack(enc.fit_transform(adult.values[:,-1]))

    return x,y

#
# x, y = read_adult()

x, y = read_adult_onehot()

x = preprocessing.scale(x)

data = model_selection.train_test_split(x, y, train_size=0.8)
x_train, x_test, y_train, y_test = data


model = keras.Sequential([
    keras.layers.Input(shape=x[0].shape),
    keras.layers.Dense(10, activation='sigmoid'),
    keras.layers.Dense(1, activation='sigmoid')

])

model.compile(optimizer=keras.optimizers.SGD(0.1),
              loss=keras.losses.binary_crossentropy,
              metrics=['acc'])

# 모델 훈련
model.fit(x_train, y_train, epochs = 50,verbose = 1) # 순전파 + 최적화함수 + 역전파

print('mae :', model.evaluate(x_test, y_test, verbose=0))
