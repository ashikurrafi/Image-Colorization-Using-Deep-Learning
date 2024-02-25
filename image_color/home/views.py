import os
import numpy as np
from django.shortcuts import render, redirect, HttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from keras.models import load_model
from PIL import Image

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


def loading_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except OSError as e:
        print(f"OS error: {e}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        try:
            image_file = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(image_file.name, image_file)
            uploaded_image_path = fs.path(filename)
            return colorize_image(request, uploaded_image_path)
        except Exception as e:
            print(f"Error uploading image: {e}")
            return render(request, 'colorizer/error.html', {'message': 'Error uploading image'})
    else:
        return render(request, 'colorizer/upload_form.html')


def colorize_image(request, image_path):
    try:
        model_path = os.path.join(settings.MODEL_ROOT, 'model.h5')
        model = loading_model(model_path)
        if model:
            # Load the uploaded image
            img = Image.open(image_path)
            img = img.resize((256, 256))  # Resize if necessary
            img_array = np.array(img) / 255.0  # Normalize pixel values
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Colorize the image
            colorized_img = model.predict(img_array)

            # Post-process the colorized image
            colorized_img = np.squeeze(colorized_img, axis=0)  # Remove batch dimension
            colorized_img = (colorized_img * 255).astype(np.uint8)  # Convert back to uint8

            # Save the colorized image
            colorized_image = Image.fromarray(colorized_img)
            fs = FileSystemStorage()
            colorized_filename = fs.save('colorized_' + os.path.basename(image_path), colorized_image)

            # Return the colorized image path for display
            colorized_image_url = fs.url(colorized_filename)
            return render(request, 'colorizer/result.html', {'colorized_image_url': colorized_image_url})
        else:
            return render(request, 'colorizer/error.html', {'message': 'Error loading model'})
    except Exception as e:
        print(f"Error colorizing image: {e}")
        return render(request, 'colorizer/error.html', {'message': 'Error colorizing image'})
