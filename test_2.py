import os
import numpy as np
from tensorflow.keras.utils import img_to_array, load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2lab
from tensorflow.keras.utils import Sequence


from keras.models import Sequential
from keras.layers import Conv2D, UpSampling2D
from keras.optimizers import RMSprop


from keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt


class DataGenerator(Sequence):
    def __init__(self, directory, batch_size, img_size=(256, 256)):
        self.directory = directory
        self.batch_size = batch_size
        self.img_size = img_size
        self.file_list = os.listdir(directory)
        self.datagen = ImageDataGenerator(
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=20,
            horizontal_flip=True
        )

    def __len__(self):
        return int(np.ceil(len(self.file_list) / self.batch_size))

    def __getitem__(self, idx):
        batch_files = self.file_list[idx *
                                     self.batch_size:(idx + 1) * self.batch_size]
        batch_images = []

        for filename in batch_files:
            img = load_img(os.path.join(self.directory, filename),
                           target_size=self.img_size)
            img_array = img_to_array(img)
            batch_images.append(img_array)

        batch_images = np.array(batch_images, dtype=float) / 255.0
        lab_batch = rgb2lab(batch_images)
        X_batch = lab_batch[:, :, :, 0]
        Y_batch = lab_batch[:, :, :, 1:] / 128.0

        return X_batch.reshape(X_batch.shape + (1,)), Y_batch


# CNN model
model = Sequential([
    Conv2D(64, (3, 3), input_shape=(256, 256, 1),
           activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same', strides=2),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same', strides=2),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    Conv2D(256, (3, 3), activation='relu', padding='same', strides=2),
    Conv2D(512, (3, 3), activation='relu', padding='same'),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    Conv2D(2, (3, 3), activation='tanh', padding='same'),
    UpSampling2D((2, 2))
])


# Compile the model
model.compile(optimizer=RMSprop(), loss='mse', metrics=['accuracy'])


# Set paths
train_data_dir = 'Dataset/Train/'
test_data_dir = 'Dataset/Test/'
save_model_dir = 'Dataset/Model/'


# Parameters
# batch_size = 10
batch_size = 100
epochs = 500


# Initialize data generators
train_generator = DataGenerator(train_data_dir, batch_size)


# Set up callbacks
tensorboard = TensorBoard(log_dir="Dataset/output/beta_run")
checkpoint = ModelCheckpoint(filepath=os.path.join(
    save_model_dir, 'model-{epoch:02d}.h5'), save_weights_only=True, period=10)


# Train the model using the generator
history = model.fit(train_generator, epochs=epochs,
                    callbacks=[tensorboard, checkpoint])


# Summarize history for model accuracy
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()


# Summarize history for model loss
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()


# Save model architecture and weights
model_json = model.to_json()
with open(os.path.join(save_model_dir, "model.json"), "w") as json_file:
    json_file.write(model_json)
model.save_weights(os.path.join(save_model_dir, "model.h5"))


# Load json and create model
json_file = open('Dataset/Model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# Load weights into new model
loaded_model.load_weights("Dataset/Model/model.h5")


loaded_model.summary()


# Test images
loaded_model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
Xtest = rgb2lab(1.0 / 255 * X[split:])[:, :, :, 0]
Xtest = Xtest.reshape(Xtest.shape + (1,))
Ytest = rgb2lab(1.0 / 255 * X[split:])[:, :, :, 1:]
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
colorize = rgb2lab(1.0 / 255 * colorize)[:, :, :, 0]
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

plt.show()
