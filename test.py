# Importing Libraries

import cv2
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dense, Dropout, Flatten, InputLayer
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.utils import array_to_img, img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf


# Getting Images

X = []
for imagename in os.listdir('Dataset/Train/'):
    X.append(img_to_array(
        load_img('Dataset/Train/'+imagename, target_size=(256, 256))))
X = np.array(X, dtype=float)


# Set up train and test data
split = int(0.95*len(X))
Xtrain = X[:split]
Xtrain = 1.0/255*Xtrain

# set up Test data
Xtest = X[split:]
Xtest = 1.0/255*Xtest


# CNN model


model = Sequential()

# Input Layer
model.add(Conv2D(64, (3, 3), input_shape=(256, 256, 1),
          activation='relu', padding='same'))

# Hidden Layers
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))

# Compiling the CNN
model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])


# Image transformer
datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    horizontal_flip=True)

# Generate training data
batch_size = 10


def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:, :, :, 0]
        Y_batch = lab_batch[:, :, :, 1:] / 128
        yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)


# Train model
tensorboard = TensorBoard(log_dir="Dataset/output/beta_run")
trainedmodel = model.fit(image_a_b_gen(batch_size), callbacks=[
                         tensorboard], epochs=500, steps_per_epoch=30)


# Summarize history for model accuracy
plt.plot(trainedmodel.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for model loss
plt.plot(trainedmodel.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Save model

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")


# load json and create model

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")


loaded_model.summary()


# Test images
loaded_model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
Xtest = rgb2lab(1.0/255*X[split:])[:, :, :, 0]
Xtest = Xtest.reshape(Xtest.shape+(1,))
Ytest = rgb2lab(1.0/255*X[split:])[:, :, :, 1:]
Ytest = Ytest / 128
print(loaded_model.evaluate(Xtest, Ytest, batch_size=10))


fig, ax = plt.subplots(24, 2, figsize=(16, 100))
row = 0
colorize = []

print('Output of the Model')


for filename in os.listdir('Dataset/Test/'):
    img = cv2.imread('Dataset/Test/' + filename)

    # Check if image is read correctly
    if img is None:
        print(f"Couldn't read image {filename}. Skipping.")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (256, 256))
    colorize.append(img_resized)

    ax[row, 0].imshow(cv2.cvtColor(
        img_resized, cv2.COLOR_BGR2RGB), interpolation='nearest')
    row += 1

colorize = np.array(colorize, dtype=float)
colorize = rgb2lab(1.0/255 * colorize)[:, :, :, 0]
colorize = colorize.reshape(colorize.shape + (1,))

# Test model
output = loaded_model.predict(colorize)
output *= 128

row = 0

# Output colorizations
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:, :, 0] = colorize[i][:, :, 0]
    cur[:, :, 1:] = output[i]
    resImage = lab2rgb(cur)

    ax[row, 1].imshow(resImage, interpolation='nearest')
    row += 1
