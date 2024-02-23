from django.shortcuts import render, redirect
from .models import Image  # Replace with your model name
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import numpy as np


def colorize_image(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        if image_file:
            # Load image using Pillow (PIL fork)
            img = Image.open(image_file)

            # Preprocess image for model input (resize, normalize, etc.)
            img = img.resize((224, 224))  # Adjust based on your model's input size
            img_array = np.array(img)
            img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Load colorization model
            model = load_model('model.h5')

            # Make prediction
            colored_image = model.predict(img_array)[0]

            # Convert back to PIL Image and prepare for display
            colored_image = np.squeeze(colored_image) * 255.0  # Denormalize
            colored_image = colored_image.astype(np.uint8)
            colored_img = Image.fromarray(colored_image)

            # Save colored image (optional)
            colored_img.save('colored_image.jpg')

            # Return success message and colored image URL
            return render(request, 'success.html', {'colored_image_url': '/static/colored_image.jpg'})
        else:
            return render(request, 'index.html', {'error': 'Please upload an image.'})
    else:
        return render(request, 'index.html')
