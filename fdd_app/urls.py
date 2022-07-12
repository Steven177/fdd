from django.urls import path

from . import views

from .views import *

urlpatterns = [
    path('', views.input, name='input'),
    path('user', views.user, name='user'),
    path('failure', views.failure, name='failure'),
    path('failure_book', views.failure_book, name='failure_book'),
    path('persona', views.persona, name='persona')
]

