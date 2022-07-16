from django.urls import path

from . import views

from .views import *

urlpatterns = [
    path('samples', views.samples, name='samples'),
    path('sample=<int:sample_id>', views.sample, name='sample'),
    path('failure_book', views.failure_book, name='failure_book'),
    path('user_scenario', views.user_scenario, name='user_scenario')
]

