from django.urls import path

from . import views

from .views import *

urlpatterns = [
    path('', views.open_model_exploration, name='open_model_exploration'),
    path('failure_book', views.failure_book, name='failure_book')
]

