from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='home'),
    # path('colorize-image/', views.colorize, name='colorize_image'),
]