# Autoencoder architecture - STEEL Dataset.
# Currently work in progress.
# Import needed files.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from os import listdir
from os.path import isfile, join
from sklearn import metrics, mixture
import imageio.v2 as imageio
import SteelDataset as sd


# Shows a series of images to visual inspect the reconstructions.
def show_data(X, n=10, title="", cmap='viridis'):
    plt.figure(figsize=(15, 5), dpi=80)
    for i in range(n):
        ax = plt.subplot(2,n,i+1)
        plt.imshow(tf.keras.utils.array_to_img(X[i]), cmap=plt.get_cmap(cmap))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.suptitle(title, fontsize = 20)
    
# Calculate features for mean, max, and standard deviation for each reconstructed image.
def error_features(reconstruction_error):
    errors = []
    for i in range(len(reconstruction_error)):
        errors += [[np.mean(reconstruction_error[i]), 
                    np.max(reconstruction_error[i]),
                    np.std(reconstruction_error[i])]]
    return errors



train_num = 3000
num_segmented = 50
training_data = sd.STEELDataAcquisition('TRAIN', train_num, num_segmented)

train_pos_des = './datasets/STEEL/STEEL_TRAINING/POSITIVE'
train_neg_des = './datasets/STEEL/STEEL_TRAINING/NEGATIVE'



neg_data = []
for extension in os.listdir(train_neg_des)[0:300]:
    neg_data += [imageio.imread(train_neg_des + "/" + extension)]



pos_data = []
for extension in os.listdir(train_neg_des)[0:200]:
    pos_data += [imageio.imread(train_neg_des + "/" + extension)]



neg_data = np.asarray(neg_data)
print(neg_data.shape)



pos_data = np.asarray(pos_data)
print(pos_data.shape)



# Keras preprocessing data augmentation layers.
data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomContrast(0.5),
    layers.experimental.preprocessing.RandomTranslation(0.3, 0.3, 'wrap')
])

# Model Architecture
input_layer = Input(shape=(256, 1600, 3), name="INPUT")
x = data_augmentation(input_layer)
# Encoding
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)

# Dense latent factors.
x = tf.keras.layers.Flatten()(x)
latent = tf.keras.layers.Dense(32)(x)
x = tf.keras.layers.Dense(32*200*8,activation='relu')(latent)
x = tf.keras.layers.Reshape((32,200,8))(x)

# Decoding
x = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2,2))(x)
output_layer = Conv2D(3, (3, 3), padding='same', name="OUTPUT")(x)

autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()


# Fits the model to the dataset.
earlystopping = EarlyStopping(monitor='loss', patience=5, min_delta=1)
autoencoder.fit(neg_data[0:100], neg_data[0:100], epochs=100, steps_per_epoch=10, batch_size=4, callbacks=[earlystopping])


pos_reconstructed = autoencoder.predict(pos_data[0:10])


neg_reconstructed = autoencoder.predict(neg_data[100:200])


# Visually compare images
show_data(pos_data[0:10], title="Original Image", n=5)
show_data(pos_reconstructed, title="Reconstructed Image", n=5)
recon_error=(np.abs(pos_data[0:10]-pos_reconstructed))
show_data(recon_error, title="Reconstruction Error", n=5)

# Visually compare images
show_data(neg_data[100:110], title="Original Image", n=5)
show_data(neg_reconstructed, title="Reconstructed Image", n=5)
recon_error=(np.abs(neg_data[100:110]-pos_reconstructed))
show_data(recon_error, title="Reconstruction Error", n=5)