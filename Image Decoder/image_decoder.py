#Modules
import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

#Data loading
from tensorflow.keras.datasets import mnist

(X_train, y_train),(X_test, y_test) = mnist.load_data()



#Selecting image
i = random.randint(1,60000)
plt.imshow( X_train[i] , cmap = 'gray')

label = y_train[i]


#Adding noise to the images

#Normalizing
X_train = X_train / 255
X_test = X_test / 255

added_noise = np.random.randn(*(28,28))

noise_factor = 0.3
added_noise = noise_factor * np.random.randn(*(28,28))

plt.imshow(added_noise)


#Adding noise to sample image
noise_factor = 0.2
sample_image = X_train[101]
noisy_sample_image = sample_image + noise_factor * np.random.randn(*(28,28))

plt.imshow(noisy_sample_image, cmap="gray")


noisy_sample_image.max()
noisy_sample_image.min()


noisy_sample_image = np.clip(noisy_sample_image, 0., 1.)


noisy_sample_image.max()
noisy_sample_image.min()

plt.imshow(noisy_sample_image, cmap="gray")


#Adding noise to all images

X_train_noisy = []
noise_factor = 0.2

for sample_image in X_train:
  sample_image_noisy = sample_image + noise_factor * np.random.randn(*(28,28))
  sample_image_noisy = np.clip(sample_image_noisy, 0., 1.)
  X_train_noisy.append(sample_image_noisy)


#Convirtiendo Set de Datos de Entrenamiento en matrriz
X_train_noisy = np.array(X_train_noisy)

plt.imshow(X_train_noisy[25], cmap="gray")



X_test_noisy = []
noise_factor = 0.4

for sample_image in X_test:
  sample_image_noisy = sample_image + noise_factor * np.random.randn(*(28,28))
  sample_image_noisy = np.clip(sample_image_noisy, 0., 1.)
  X_test_noisy.append(sample_image_noisy)

X_test_noisy = np.array(X_test_noisy)



X_test_noisy.shape
plt.imshow(X_test_noisy[10], cmap = 'gray')



#Bulding model
autoencoder = tf.keras.models.Sequential()

#Convolutional Layers
#1st convolutional layer
autoencoder.add(tf.keras.layers.Conv2D(16, (3,3), strides=1, padding="same", input_shape=(28, 28, 1)))
autoencoder.add(tf.keras.layers.MaxPooling2D((2,2), padding="same"))

#2st convolutional layer
autoencoder.add(tf.keras.layers.Conv2D(8, (3,3), strides=1, padding="same"))
autoencoder.add(tf.keras.layers.MaxPooling2D((2,2), padding="same"))

#Building decoding
#1st decoding layer
autoencoder.add(tf.keras.layers.Conv2D(8, (3,3), strides=1, padding="same"))

#Building decoder
autoencoder.add(tf.keras.layers.UpSampling2D((2, 2)))
autoencoder.add(tf.keras.layers.Conv2DTranspose(8,(3,3), strides=1, padding="same"))


autoencoder.add(tf.keras.layers.UpSampling2D((2, 2)))
autoencoder.add(tf.keras.layers.Conv2DTranspose(1, (3,3), strides=1, activation='sigmoid', padding="same"))

#Compiling model
autoencoder.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001))

autoencoder.summary()


#Training model
autoencoder.fit(X_train_noisy.reshape(-1, 28, 28, 1),          
          X_train.reshape(-1, 28, 28, 1), 
          epochs=10, 
          batch_size=200)


#Testing model
denoised_images = autoencoder.predict(X_test_noisy[:15].reshape(-1, 28, 28, 1))
denoised_images.shape

fig, axes = plt.subplots(nrows=2, ncols=15, figsize=(30,6))
for images, row in zip([X_test_noisy[:15], denoised_images], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((28, 28)), cmap='gray')























