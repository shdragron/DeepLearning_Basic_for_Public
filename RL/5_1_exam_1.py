# 5_1_exam_1.py
import keras
import numpy as np


# 통과 기준
# mae 오차 0.0001 이하
class EpochStep(keras.callbacks.Callback):
    def __init__(self, step=500):
        super().__init__()
        self.step = step

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        if epoch % self.step == 0:
            print(epoch, logs)
            # print('{:5} : acc {:.03f} loss {:.03f}'.format(epoch, logs['acc'], logs['loss']))


x = [0, 1, 2, 3, 4, 5, 6]
y = [-3, -2, -1, 0, 1, 2, 3]

x = np.reshape(x, newshape=[-1, 1])     # (7,) -> (7, 1)
y = np.reshape(y, newshape=[-1, 1])

model = keras.Sequential([
    keras.layers.Dense(1)
])

model.compile(optimizer=keras.optimizers.Adam(0.01),
              loss=keras.losses.mean_squared_error,
              metrics=['mae'])

x_test = np.reshape([25, 30, 35, 40, 45], newshape=(-1, 1))
y_test = np.reshape([22, 27, 32, 37, 42], newshape=(-1, 1))

model.fit(x, y, epochs=1000, verbose=0,
          validation_data=(x_test, y_test),
          callbacks=[EpochStep(100)])

p = model.predict(x_test, verbose=0)
print('mae :', np.mean(np.abs(p - y_test)))
print('mae :', model.evaluate(x_test, y_test, verbose=0))

