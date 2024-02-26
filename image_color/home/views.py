import os

import cv2
import numpy as np
from django.shortcuts import render, redirect, HttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from keras.models import load_model
from PIL import Image
import sys

sys.path.append('D:/8th SEM/IC_project/Image-Colorization-Using-Deep-Learning/image_color/home/colimg')
from colorize_img import colorize_image

import h5py


def inspect_hdf5(request):
    # Define the path to the HDF5 file
    model_path = os.path.join(settings.BASE_DIR, 'home', 'models', 'model.h5')

    try:
        # Open the HDF5 file in read-only mode
        with h5py.File(model_path, 'r') as file:
            # List all groups and datasets in the file
            file_content = list(file.keys())

            # Attempt to access a specific dataset
            dataset_name = 'dataset_name'
            dataset_content = None
            if dataset_name in file:
                dataset_content = file[dataset_name][()]
            else:
                dataset_content = f"Dataset '{dataset_name}' not found in the file."

        # Render a template with the HDF5 file content
        return render(request, 'inspect_hdf5.html', {'file_content': file_content, 'dataset_content': dataset_content})

    except Exception as e:
        # Handle errors, e.g., if the file cannot be opened
        return HttpResponse(f"Error: {e}")


# def loading_model(model_path):
#     try:
#         model = load_model(model_path)
#         return model
#     except OSError as e:
#         print(f"OS error: {e}")
#         return None
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         return None


# def upload_image(request):
#     if request.method == 'POST' and request.FILES['image']:
#         try:
#             image_file = request.FILES['image']
#             fs = FileSystemStorage()
#             filename = fs.save(image_file.name, image_file)
#             uploaded_image_path = fs.path(filename)
#             return colorize_image(request, uploaded_image_path)
#         except Exception as e:
#             print(f"Error uploading image: {e}")
#             return render(request, 'colorizer/error.html', {'message': 'Error uploading image'})
#     else:
#         return render(request, 'colorizer/upload_form.html')

def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        try:
            image_file = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(image_file.name, image_file)
            uploaded_image_path = fs.path(filename)
            colorized_image = colorize_image(uploaded_image_path)
            if colorized_image is not None:
                output_folder = os.path.join(settings.MEDIA_ROOT, 'colorized_images')
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                output_image_path = os.path.join(output_folder, 'colorized_image.jpg')
                cv2.imwrite(output_image_path, cv2.cvtColor(colorized_image, cv2.COLOR_RGB2BGR))
                colorized_image_url = fs.url(output_image_path)
                return render(request, 'colorizer/result.html', {'colorized_image_url': colorized_image_url})
            else:
                return render(request, 'colorizer/error.html', {'message': 'Error colorizing image'})
        except Exception as e:
            print(f"Error uploading image: {e}")
            return render(request, 'colorizer/error.html', {'message': 'Error uploading image'})
    else:
        return render(request, 'colorizer/upload_form.html')

# def colorizing_image(request, image_path):
#     try:
#         model_path = os.path.join(settings.MODEL_ROOT, 'model.h5')
#
#         colorized_image = colorize_image(image_path, model_path)  # Corrected function call
#
#         if colorized_image:
#             # Save the colorized image
#             fs = FileSystemStorage()
#             colorized_filename = fs.save('colorized_' + os.path.basename(image_path), colorized_image)
#
#             # Return the colorized image path for display
#             colorized_image_url = fs.url(colorized_filename)
#             return render(request, 'colorizer/result.html', {'colorized_image_url': colorized_image_url})
#         else:
#             return render(request, 'colorizer/error.html', {'message': 'Error colorizing image'})
#
#     except Exception as e:
#         print(f"Error colorizing image: {e}")
#         return render(request, 'colorizer/error.html', {'message': 'Error colorizing image'})

# def colorizing_image(request, image_path):
#     try:
#         try:
#             # Define the paths to the model JSON file, model weights file, and output folder
#             model_json_path = settings.MODEL_JSON_PATH
#             model_weights_path = settings.MODEL_WEIGHTS_PATH
#             output_folder = settings.COLORIZED_IMAGES_DIR
#
#             # Call the colorize_image function with the appropriate arguments
#             colorize_image(image_path, model_json_path, model_weights_path, output_folder)
#
#         if colorized_image_path:
#             # Return the colorized image path for display
#             return render(request, 'colorizer/result.html', {'colorized_image_path': colorized_image_path})
#         else:
#             return render(request, 'colorizer/error.html', {'message': 'Error colorizing image'})
#
#     except Exception as e:
#         print(f"Error colorizing image: {e}")
#         return render(request, 'colorizer/error.html', {'message': 'Error colorizing image'})

# def colorizing_image(request, image_path):
#     try:
#         # Call the colorize_image function with the appropriate arguments
#         model_json_path = settings.MODEL_JSON_PATH
#         model_weights_path = settings.MODEL_WEIGHTS_PATH
#         output_folder = settings.COLORIZED_IMAGES_DIR
#
#         # Call the colorize_image function with the appropriate arguments
#
#         colorized_image_path = colorize_image(image_path, model_json_path, model_weights_path, output_folder)
#
#         if colorized_image_path:
#             # Return the colorized image path for display
#             return render(request, 'colorizer/result.html', {'colorized_image_path': colorized_image_path})
#         else:
#             return render(request, 'colorizer/error.html', {'message': 'Error colorizing image'})
#
#     except Exception as e:
#         print(f"Error colorizing image: {e}")
#         return render(request, 'colorizer/error.html', {'message': 'Error colorizing image'})

def colorizing_image(request, image_path):
    try:
        model_json_path = os.path.join(settings.MODEL_ROOT, 'model.json')  # Path to model JSON file
        model_weights_path = os.path.join(settings.MODEL_ROOT, 'model.h5')  # Path to model weights file
        output_folder = os.path.join(settings.MEDIA_ROOT, 'colorized_images')  # Output folder for colorized images

        # Call the colorize_image function with the appropriate arguments
        colorize_image(image_path, model_json_path, model_weights_path, output_folder)

        # Assuming the colorized image has been saved with a fixed name 'colorized_image.jpg'
        colorized_image_path = os.path.join(output_folder, 'colorized_image.jpg')

        if os.path.exists(colorized_image_path):
            # Return the colorized image path for display
            return render(request, 'colorizer/result.html', {'colorized_image_path': colorized_image_path})
        else:
            return render(request, 'colorizer/error.html', {'message': 'Error colorizing image'})

    except Exception as e:
        print(f"Error colorizing image: {e}")
        return render(request, 'colorizer/error.html', {'message': 'Error colorizing image'})