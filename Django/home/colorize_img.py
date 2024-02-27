import os
import cv2
import numpy as np
from keras.models import model_from_json
from PIL import Image
from skimage.color import lab2rgb, rgb2lab


def colorize_image(image_path, model_json_path, model_weights_path):
    try:
        # Load the pre-trained model
        with open(model_json_path, 'r') as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(model_weights_path)

        # Load the uploaded image
        img = Image.open(image_path)
        img = img.resize((256, 256))  # Resize if necessary
        img_array = np.array(img)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img_array = img_array / 255.0  # Normalize pixel values
        img_array_gray = rgb2lab(img_array)[:, :, 0]
        img_array_gray = img_array_gray.reshape(1, img_array_gray.shape[0], img_array_gray.shape[1], 1)

        # Predict colorization
        output = loaded_model.predict(img_array_gray)
        output *= 128

        # Combine grayscale image and predicted ab values
        colorized_img = np.zeros((256, 256, 3))
        colorized_img[:, :, 0] = img_array_gray[0][:, :, 0]
        colorized_img[:, :, 1:] = output[0]
        colorized_img = lab2rgb(colorized_img)

        # Convert colorized image to PIL format
        colorized_img = (colorized_img * 255).astype(np.uint8)
        colorized_image = Image.fromarray(colorized_img)

        return colorized_image

    except Exception as e:
        print(f"Error colorizing image: {e}")
        return None
