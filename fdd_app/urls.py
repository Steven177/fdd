from django.urls import path

from . import views

from .views import *

urlpatterns = [
    path('create_persona', views.create_persona, name='create_persona'),
    path('create_scenario', views.create_scenario, name='create_scenario'),
    path('samples', views.samples, name='samples'),
    path('sample=<int:sample_id>', views.sample, name='sample'),
    path('failure_book', views.failure_book, name='failure_book'),
]

