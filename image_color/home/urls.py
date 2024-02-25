from django.urls import path
from . import views

urlpatterns = [
    # path('', views.index, name='home'),
    # path('', views.colorize_image, name='colorize_image'),
    path('upload/', views.upload_image, name='upload_image'),
    path('result/', views.colorize_image, name='colorize_image'),
    path('inspect-hdf5/', views.inspect_hdf5, name='inspect_hdf5'),
]