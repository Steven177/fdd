from django.urls import path

from . import views

from .views import *

urlpatterns = [
    path('create_persona', views.create_persona, name='create_persona'),

    path('persona=<int:persona_id>/update_persona', views.update_persona, name='update_persona'),
    path('persona=<int:persona_id>/delete_persona', views.delete_persona, name='delete_persona'),
    path('persona=<int:persona_id>/create_scenarios', views.create_scenarios, name='create_scenarios'),

    path('persona=<int:persona_id>/upload/', views.file_upload, name='file_upload'),
    path('persona=<int:persona_id>/scenario=<int:scenario_id>/sample=<int:sample_id>/augmentations', views.augmentations, name='augmentations'),

    path('persona=<int:persona_id>/scenario=<int:scenario_id>/update_scenario', views.update_scenario, name='update_scenario'),
    path('persona=<int:persona_id>/scenario=<int:scenario_id>/delete_scenario', views.delete_scenario, name='delete_scenario'),

    path('persona=<int:persona_id>/scenario=<int:scenario_id>/samples', views.samples, name='samples'),
    path('persona=<int:persona_id>/scenario=<int:scenario_id>/sample=<int:sample_id>/read_sample', views.read_sample, name='read_sample'),
    path('persona=<int:persona_id>/scenario=<int:scenario_id>/sample=<int:sample_id>/update_sample', views.update_sample, name='update_sample'),
    path('persona=<int:persona_id>/scenario=<int:scenario_id>/sample=<int:sample_id>/delete_sample', views.delete_sample, name='delete_sample'),
    path('persona=<int:persona_id>/scenario=<int:scenario_id>/sample=<int:sample_id>/failure_exploration', views.failure_exploration, name='failure_exploration'),

    path('persona=<int:persona_id>/failure_analysis', views.failure_analysis, name='failure_analysis'),
    path('read_ai', views.read_ai, name='read_ai'),

    path('data', views.data, name='data')

]

