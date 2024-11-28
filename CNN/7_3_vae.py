# 7_3_vae.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

VAE = __import__('7_2_vae')

_, x_test, _, y_test = VAE.load_fashion_mnist()
x_test = VAE.preprocess(x_test)

encoder, decoder, vae = VAE.make_vae()

vae.load_weights('./models/vae-33-131.39.weights.h5')

# --------------------------- #
test_images = x_test[:5000]
test_labels = y_test[:5000]

z_mean, z_log_var, reconstruction = vae.predict(test_images, verbose=0)
# VAE.show_digit_double(test_images, reconstruction)

# --------------------------- #
z_mean, z_log_var, embeddings = encoder.predict(test_images, verbose=0)
print(embeddings[:10])
print(embeddings.shape)

# plt.figure(figsize=(8, 8))
# plt.scatter(embeddings[:, 0], embeddings[:, 1], cmap='rainbow', c=test_labels, alpha=0.5, s=3)
# plt.colorbar()
# plt.show()

# --------------------------- #
grid_width, grid_height = 6, 3
sample = np.random.normal(size=(grid_width * grid_height, 2))

# 누적확률분포
p = norm.cdf(embeddings)
p_sample = norm.cdf(sample)

# plt.figure(figsize=(8, 8))
# plt.scatter(embeddings[:, 0], embeddings[:, 1], cmap='rainbow', c=test_labels, alpha=0.5, s=3)
# plt.scatter(sample[:, 0], sample[:, 1], c='#00B0E0', s=40)
# plt.colorbar()
# plt.show()

reconstructions = decoder.predict(sample)

# fig = plt.figure(figsize=(8, grid_height * 2))
# for i in range(grid_width * grid_height):
#     ax = fig.add_subplot(grid_height, grid_width, i+1)
#     ax.axis('off')
#     ax.text(0.5, -0.35, str(np.round(sample[i, :], 1)),
#             fontsize=10, ha='center', transform=ax.transAxes)
#     ax.imshow(reconstructions[i, :, :], cmap='Greys')
#
# plt.tight_layout()
# plt.show()

# --------------------------- #
x = norm.ppf(np.linspace(0, 1, 15))
y = norm.ppf(np.linspace(1, 0, 15))

xv, yv = np.meshgrid(x, y)
xv = xv.flatten()
yv = yv.flatten()
grid = np.array(list(zip(xv, yv)))

# plt.scatter(grid[:, 0], grid[:, 1], c='black', s=10)
# plt.show()

# --------------------------- #
# grid를 잠재벡터로 간주
reconstructions = decoder.predict(grid)

fig = plt.figure(figsize=(12, 12))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(15 ** 2):
    ax = fig.add_subplot(15, 15, i+1)
    ax.axis('off')
    ax.imshow(reconstructions[i, :, :], cmap='Greys')

plt.tight_layout()
plt.show()
