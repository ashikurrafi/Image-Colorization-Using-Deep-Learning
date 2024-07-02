# from django.conf import settings
# from django.http import HttpResponse
# from django.shortcuts import render
# import os
# import cv2
# import numpy as np
# from skimage.color import rgb2lab, lab2rgb
# from keras.models import model_from_json
# from django.core.files.storage import FileSystemStorage

# BASE_DIR = settings.BASE_DIR

# # Create your views here.
# def index(request):
#     return HttpResponse("Hello world!")

# # def colorize_image(image_path, image_name):
# #     # Load the pre-trained model
# #     model_json_path = 'icolor/home/dl_models/model.json'
# #     model_weights_path = 'icolor/home/dl_models/model.h5'

# #     json_file = open(model_json_path, 'r')
# #     loaded_model_json = json_file.read()
# #     json_file.close()
# #     loaded_model = model_from_json(loaded_model_json)
# #     # load weights into new model
# #     loaded_model.load_weights(model_weights_path)
# #     loaded_model.summary()

# #     # Load the image
# #     img = cv2.imread(image_path)

# #     # Check if image is read correctly
# #     if img is None:
# #         print(f"Couldn't read image {image_path}. Exiting.")
# #         return

# #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #     img_resized = cv2.resize(img, (256, 256))

# #     # Preprocess the image for colorization
# #     colorize = np.array(img_resized, dtype=float)
# #     colorize = rgb2lab(1.0/255 * colorize)[:, :, 0]
# #     # Reshape with an extra dimension
# #     colorize = colorize.reshape(1, colorize.shape[0], colorize.shape[1], 1)

# #     # Predict colorization
# #     output = loaded_model.predict(colorize)
# #     output *= 128

# #     # Combine grayscale image and predicted ab values
# #     cur = np.zeros((256, 256, 3))
# #     cur[:, :, 0] = colorize[0][:, :, 0]
# #     cur[:, :, 1:] = output[0]
# #     resImage = lab2rgb(cur)

# #     # Convert the colorized image to the appropriate data type for saving
# #     resImage_uint8 = (resImage * 255).astype(np.uint8)

# #     # Save the colorized image
# #     output_folder = os.path.dirname(image_path)
# #     output_image_name = f"colorized_{image_name}"
# #     output_image_path = os.path.join(output_folder, output_image_name)
# #     cv2.imwrite(output_image_path, cv2.cvtColor(resImage_uint8, cv2.COLOR_RGB2BGR))
# #     print(f"Colorized image saved at: {output_image_path}")


# def upload_image(request):
#     if request.method == 'POST' and request.FILES['image']:
#         try:
#             image_file = request.FILES['image']
#             fs = FileSystemStorage()
#             filename = fs.save(image_file.name, image_file)
#             uploaded_image_path = fs.path(filename)

#             # Get the image name
#             image_name = os.path.basename(uploaded_image_path)

#             # Colorize the uploaded image
#             colorize_image(uploaded_image_path, image_name)

#             # Provide the colorized image URL to the template for rendering
#             colorized_image_url = fs.url(f"colorized_{image_name}")
#             original_image_url = fs.url(filename)  # Get the URL of the original image

#             # Define the download filename
#             download_filename = f"colorized_{image_name}"

#             return render(request, 'colorizer/result.html',
#                           {'original_image_url': original_image_url, 'colorized_image_url': colorized_image_url,
#                            'download_filename': download_filename})
#         except Exception as e:
#             print(f"Error uploading image: {e}")
#             return render(request, 'colorizer/error.html', {'message': 'Error uploading image'})
#     else:
#         return render(request, 'colorizer/upload_form.html')

# def colorize_image(image_path, image_name):
#     # Load the pre-trained model
#     model_json_path = os.path.join(BASE_DIR, 'home', 'dl_models', 'model.json')
#     model_weights_path = os.path.join(BASE_DIR, 'home', 'dl_models', 'model.h5')

#     json_file = open(model_json_path, 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     loaded_model = model_from_json(loaded_model_json)
#     loaded_model.load_weights(model_weights_path)
#     loaded_model.summary()

#     # Load the image
#     img = cv2.imread(image_path)

#     # Check if image is read correctly
#     if img is None:
#         print(f"Couldn't read image {image_path}. Exiting.")
#         return

#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_resized = cv2.resize(img, (256, 256))

#     # Preprocess the image for colorization
#     colorize_input = np.array(img_resized, dtype=float)
#     colorize_input = rgb2lab(1.0/255 * colorize_input)[:, :, 0]
#     colorize_input = colorize_input.reshape(1, colorize_input.shape[0], colorize_input.shape[1], 1)

#     # Predict colorization
#     output = loaded_model.predict(colorize_input)
#     output *= 128

#     # Combine grayscale image and predicted ab values
#     cur = np.zeros((256, 256, 3))
#     cur[:, :, 0] = colorize_input[0][:, :, 0]
#     cur[:, :, 1:] = output[0]
#     colorized_image = lab2rgb(cur)

#     # Convert the colorized image to the appropriate data type for saving
#     colorized_image_uint8 = (colorized_image * 255).astype(np.uint8)

#     # Save the colorized image
#     output_folder = os.path.dirname(image_path)
#     output_image_name = f"colorized_{image_name}"
#     output_image_path = os.path.join(output_folder, output_image_name)
#     cv2.imwrite(output_image_path, cv2.cvtColor(colorized_image_uint8, cv2.COLOR_RGB2BGR))
#     print(f"Colorized image saved at: {output_image_path}")


# from django.http import HttpResponse
# from django.shortcuts import render
# import os
# import cv2
# import numpy as np
# from skimage.color import rgb2lab, lab2rgb
# from keras.models import model_from_json
# from django.core.files.storage import FileSystemStorage
# from django.conf import settings  # Import Django settings

# BASE_DIR = settings.BASE_DIR  # Use BASE_DIR from Django settings

# # Create your views here.
# def index(request):
#     return HttpResponse("Hello world!")

# def load_model(model_json_path, model_weights_path):
#     # Load the JSON model configuration
#     with open(model_json_path, 'r') as json_file:
#         loaded_model_json = json_file.read()

#     # Create model from JSON
#     loaded_model = model_from_json(loaded_model_json)

#     # Load weights into new model
#     loaded_model.load_weights(model_weights_path)

#     return loaded_model

# def colorize_image(image_path, image_name):
#     try:
#         # Define paths to model JSON and weights
#         model_json_path = os.path.join(BASE_DIR, 'home', 'dl_models', 'model.json')
#         model_weights_path = os.path.join(BASE_DIR, 'home', 'dl_models', 'model.h5')

#         # Load the pre-trained model
#         loaded_model = load_model(model_json_path, model_weights_path)
#         loaded_model.summary()

#         # Load the image
#         img = cv2.imread(image_path)

#         # Check if image is read correctly
#         if img is None:
#             print(f"Couldn't read image {image_path}. Exiting.")
#             return

#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img_resized = cv2.resize(img, (256, 256))

#         # Preprocess the image for colorization
#         colorize_input = np.array(img_resized, dtype=float)
#         colorize_input = rgb2lab(1.0/255 * colorize_input)[:, :, 0]
#         colorize_input = colorize_input.reshape(1, colorize_input.shape[0], colorize_input.shape[1], 1)

#         # Predict colorization
#         output = loaded_model.predict(colorize_input)
#         output *= 128

#         # Combine grayscale image and predicted ab values
#         cur = np.zeros((256, 256, 3))
#         cur[:, :, 0] = colorize_input[0][:, :, 0]
#         cur[:, :, 1:] = output[0]
#         colorized_image = lab2rgb(cur)

#         # Convert the colorized image to the appropriate data type for saving
#         colorized_image_uint8 = (colorized_image * 255).astype(np.uint8)

#         # Save the colorized image
#         output_folder = os.path.dirname(image_path)
#         output_image_name = f"colorized_{image_name}"
#         output_image_path = os.path.join(output_folder, output_image_name)
#         cv2.imwrite(output_image_path, cv2.cvtColor(colorized_image_uint8, cv2.COLOR_RGB2BGR))
#         print(f"Colorized image saved at: {output_image_path}")

#     except Exception as e:
#         print(f"Error processing image: {e}")

# # def upload_image(request):
# #     if request.method == 'POST' and request.FILES.get('image'):
# #         try:
# #             image_file = request.FILES['image']
# #             fs = FileSystemStorage()
# #             filename = fs.save(image_file.name, image_file)
# #             uploaded_image_path = fs.path(filename)

# #             # Get the image name
# #             image_name = os.path.basename(uploaded_image_path)

# #             # Colorize the uploaded image
# #             colorize_image(uploaded_image_path, image_name)

# #             # Provide the colorized image URL to the template for rendering
# #             colorized_image_url = fs.url(f"colorized_{image_name}")
# #             original_image_url = fs.url(filename)  # Get the URL of the original image

# #             # Define the download filename
# #             download_filename = f"colorized_{image_name}"

# #             return render(request, 'colorizer/result.html',
# #                           {'original_image_url': original_image_url, 'colorized_image_url': colorized_image_url,
# #                            'download_filename': download_filename})
# #         except KeyError:
# #             print("No image file selected.")
# #             return render(request, 'colorizer/upload_form.html', {'error_message': 'No image file selected.'})
# #         except Exception as e:
# #             print(f"Error uploading image: {e}")
# #             return render(request, 'colorizer/error.html', {'message': f'Error uploading image: {e}'})
# #     else:
# #         return render(request, 'colorizer/upload_form.html')

# def upload_image(request):
#     if request.method == 'POST' and request.FILES.get('image'):
#         try:
#             image_file = request.FILES['image']
#             fs = FileSystemStorage()
#             filename = fs.save(image_file.name, image_file)
#             uploaded_image_path = fs.path(filename)

#             # Get the image name
#             image_name = os.path.basename(uploaded_image_path)

#             # Colorize the uploaded image
#             colorize_image(uploaded_image_path, image_name)

#             # Provide the colorized image URL to the template for rendering
#             colorized_image_url = fs.url(f"colorized_{image_name}")
#             original_image_url = fs.url(filename)  # Get the URL of the original image

#             # Define the download filename
#             download_filename = f"colorized_{image_name}"

#             # Logging paths for debugging
#             print(f"Uploaded image path: {uploaded_image_path}")
#             print(f"Colorized image URL: {colorized_image_url}")
#             print(f"Original image URL: {original_image_url}")

#             return render(request, 'colorizer/result.html',
#                           {'original_image_url': original_image_url, 'colorized_image_url': colorized_image_url,
#                            'download_filename': download_filename})
#         except KeyError:
#             print("No image file selected.")
#             return render(request, 'colorizer/upload_form.html', {'error_message': 'No image file selected.'})
#         except Exception as e:
#             print(f"Error uploading image: {e}")
#             return render(request, 'colorizer/error.html', {'message': f'Error uploading image: {e}'})
#     else:
#         return render(request, 'colorizer/upload_form.html')

from django.http import HttpResponse
from django.shortcuts import render
import os
import cv2
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from keras.models import model_from_json
from django.core.files.storage import FileSystemStorage
from django.conf import settings  # Import Django settings

BASE_DIR = settings.BASE_DIR  # Use BASE_DIR from Django settings
MEDIA_ROOT = settings.MEDIA_ROOT  # Use MEDIA_ROOT from Django settings

# Create your views here.


def index(request):
    return HttpResponse("Hello world!")


def load_model(model_json_path, model_weights_path):
    # Load the JSON model configuration
    with open(model_json_path, 'r') as json_file:
        loaded_model_json = json_file.read()

    # Create model from JSON
    loaded_model = model_from_json(loaded_model_json)

    # Load weights into new model
    loaded_model.load_weights(model_weights_path)

    return loaded_model


def colorize_image(image_path, image_name):
    try:
        # Define paths to model JSON and weights
        model_json_path = os.path.join(
            BASE_DIR, 'home', 'dl_models', 'model.json')
        model_weights_path = os.path.join(
            BASE_DIR, 'home', 'dl_models', 'model.h5')

        # Load the pre-trained model
        loaded_model = load_model(model_json_path, model_weights_path)
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
        colorize_input = np.array(img_resized, dtype=float)
        colorize_input = rgb2lab(1.0/255 * colorize_input)[:, :, 0]
        colorize_input = colorize_input.reshape(
            1, colorize_input.shape[0], colorize_input.shape[1], 1)

        # colorize_input = np.array(img_resized, dtype=float)
        # colorize_input = rgb2lab(1.0/255 * colorize_input)[:, :, 0]
        # colorize_input = colorize_input.reshape(
        #     1, colorize_input.shape[0], colorize_input.shape[1], 1)

        # Predict colorization
        output = loaded_model.predict(colorize_input)
        output *= 128

        # Combine grayscale image and predicted ab values
        cur = np.zeros((256, 256, 3))
        cur[:, :, 0] = colorize_input[0][:, :, 0]
        cur[:, :, 1:] = output[0]
        colorized_image = lab2rgb(cur)

        # output = loaded_model.predict(colorize_input)
        # output *= 128

        # cur = np.zeros((256, 256, 3))
        # cur[:, :, 0] = colorize_input[0][:, :, 0]
        # cur[:, :, 1:] = output[0]
        # colorized_image = lab2rgb(cur)

        # Convert the colorized image to the appropriate data type for saving
        colorized_image_uint8 = (colorized_image * 255).astype(np.uint8)

        # Save the colorized image
        colorized_image_path = os.path.join(
            MEDIA_ROOT, 'colorized', f"colorized_{image_name}")
        cv2.imwrite(colorized_image_path, cv2.cvtColor(
            colorized_image_uint8, cv2.COLOR_RGB2BGR))
        print(f"Colorized image saved at: {colorized_image_path}")

    except Exception as e:
        print(f"Error processing image: {e}")


def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            image_file = request.FILES['image']
            fs = FileSystemStorage(
                location=os.path.join(MEDIA_ROOT, 'originals'))
            filename = fs.save(image_file.name, image_file)
            uploaded_image_path = fs.path(filename)

            # Get the image name
            image_name = os.path.basename(uploaded_image_path)

            # Colorize the uploaded image
            colorize_image(uploaded_image_path, image_name)

            # Provide the colorized image URL to the template for rendering
            colorized_image_url = os.path.join(
                settings.MEDIA_URL, 'colorized', f"colorized_{image_name}")
            # Get the URL of the original image
            original_image_url = fs.url(filename)

            # Define the download filename
            download_filename = f"colorized_{image_name}"

            return render(request, 'colorizer/result.html',
                          {'original_image_url': original_image_url, 'colorized_image_url': colorized_image_url,
                           'download_filename': download_filename})
        except KeyError:
            print("No image file selected.")
            return render(request, 'colorizer/upload_form.html', {'error_message': 'No image file selected.'})
        except Exception as e:
            print(f"Error uploading image: {e}")
            return render(request, 'colorizer/error.html', {'message': f'Error uploading image: {e}'})
    else:
        return render(request, 'colorizer/upload_form.html')
