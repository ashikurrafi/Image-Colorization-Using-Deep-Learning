from django.shortcuts import render, redirect
from .models import Image  # Replace with your model name
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import numpy as np
