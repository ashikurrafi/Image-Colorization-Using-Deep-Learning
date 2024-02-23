from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='home'),
    path('colorize-image/', views.colorize_image, name='colorize_image'),
]