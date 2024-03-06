import os
import cv2
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from keras.models import model_from_json


def colorize_image(image_path, image_name):
    # Load the pre-trained model
    model_json_path = 'Image_color_django/home/colimg/models/model.json'
    model_weights_path = 'Image_color_django/home/colimg/models/model.h5'

    json_file = open(model_json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_weights_path)
    loaded_model.summary()

    # Load the image
    img = cv2.imread(image_path)

    # Check if image is read correctly
    if img is None:
        print(f"Couldn't read image {image_path}. Exiting.")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (256, 256))

    # Preprocess the image for colorization
    colorize = np.array(img_resized, dtype=float)
    colorize = rgb2lab(1.0/255 * colorize)[:, :, 0]
    # Reshape with an extra dimension
    colorize = colorize.reshape(1, colorize.shape[0], colorize.shape[1], 1)

    # Predict colorization
    output = loaded_model.predict(colorize)
    output *= 128

    # Combine grayscale image and predicted ab values
    cur = np.zeros((256, 256, 3))
    cur[:, :, 0] = colorize[0][:, :, 0]
    cur[:, :, 1:] = output[0]
    resImage = lab2rgb(cur)

    # Convert the colorized image to the appropriate data type for saving
    resImage_uint8 = (resImage * 255).astype(np.uint8)

    # Save the colorized image
    output_folder = os.path.dirname(image_path)
    output_image_name = f"colorized_{image_name}"
    output_image_path = os.path.join(output_folder, output_image_name)
    cv2.imwrite(output_image_path, cv2.cvtColor(
        resImage_uint8, cv2.COLOR_RGB2BGR))
    print(f"Colorized image saved at: {output_image_path}")
