from django.urls import path

from . import views

from .views import *

urlpatterns = [
    # /fdd_app
    path('', views.index, name='index'),
    # /fdd_app/capabilities
    path('false_observation', views.false_observation, name="false_observation"),
    # /fdd_app/false_observation
    path('failing_to_observe', views.failing_to_observe, name="failing_to_observe"),
    path('false_observation', views.false_observation, name="false_observation"),
    path('false_observation', views.false_observation, name="false_observation"),
    path('false_observation', views.false_observation, name="false_observation"),
    # path('detail', views.detail, name='detail'),
]

