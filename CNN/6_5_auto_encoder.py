# 6_5_auto_encoder.py

import numpy as np
import matplotlib.pyplot as plt

AE = __import__('6_3_auto_encoder')
# print(AE.__name__)
_, x_test, _, y_test = AE.load_fashion_mnist()

x_test = AE.preprocess(x_test)

encoder, decoder, auto_encoder = AE.make_auto_encoder()

auto_encoder.load_weights('./models/ae--95-0.62.weights.h5')

#----------------------------------------------------------------------------------------------------------#

test_images = x_test[:5000]
test_labels = y_test[:5000]

# predictions = auto_encoder.predict(test_images, verbose=1)
# AE.show_digit_double(test_images, predictions)

#---------------------------------------------------------------------------------------------------------#

# 잠재층 안에 있는 데이터들의 분포

embeddings = encoder.predict(test_images, verbose=1)
print(embeddings[:10])
print(embeddings.shape)




# plt.figure(figsize=(8, 8))
# plt.scatter(embeddings[:, 0], embeddings[:, 1], cmap='rainbow', c=test_labels, alpha=0.5, s=3)
# plt.colorbar()
# plt.show()

#--------------------------------------------------------------------------------------------------------#

mins = np.min(embeddings, axis=0)
maxs = np.max(embeddings, axis=0)

grid_width, grid_height = 6, 3
samples = np.random.uniform(mins, maxs, (grid_width * grid_height,2))

# plt.figure(figsize=(8, 8))
# plt.scatter(embeddings[:, 0], embeddings[:, 1], cmap='rainbow', c=test_labels, alpha=0.5, s=3)
# plt.scatter(samples[:, 0], samples[:, 1], c='#00B0E0', s=40)
# plt.colorbar()
# plt.show()

reconstructions = decoder.predict(samples, verbose=1)


# fig = plt.figure(figsize=(8, grid_height * 2))
# for i in range(grid_height * grid_width):
#     ax = fig.add_subplot(grid_height, grid_width, i+1)
#     ax.axis('off')
#     ax.text(0.5, -0.35, str(np.round(samples[i,:],1)),
#             fontsize = 10, ha='center', transform=ax.transAxes)
#     ax.imshow(reconstructions[i, :, :], cmap='Greys')
#
# plt.tight_layout()
# plt.show()


#--------------------------------------------------------------------------------------------------------#

# plt.figure(figsize=(8, 8))
# plt.scatter(embeddings[:, 0], embeddings[:, 1], cmap='rainbow', c=test_labels, alpha=0.5, s=3)
# plt.colorbar()

# 눈금 만들기

x = np.linspace(min(embeddings[:,0]), max(embeddings[:,0]), 15)
y = np.linspace(min(embeddings[:,1]), max(embeddings[:,1]), 15)
print(x)

xv, yv = np.meshgrid(x, y)
xv = xv.flatten()
yv = yv.flatten()
grid = np.array(list(zip(xv, yv)))

# plt.scatter(grid[:, 0], grid[:, 1], c= 'black', s=10)
# plt.show()

#--------------------------------------------------------------------------------------------------------#

reconstruction = decoder.predict(grid, verbose=1)
fig = plt.figure(figsize = (12,12))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(15 ** 2):
    ax = fig.add_subplot(15, 15, i+1)
    ax.axis('off')
    ax.imshow(reconstruction[i,:,:], cmap='Greys')

plt.tight_layout()
plt.show()









