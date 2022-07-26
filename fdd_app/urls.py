from django.urls import path

from . import views

from .views import *

urlpatterns = [
    path('create_persona', views.create_persona, name='create_persona'),
    path('persona=<int:persona_id>/read_persona', views.read_persona, name='read_persona'),
    path('persona=<int:persona_id>/update_persona', views.update_persona, name='update_persona'),
    path('persona=<int:persona_id>/delete_persona', views.delete_persona, name='delete_persona'),

    path('persona=<int:persona_id>/create_scenario', views.create_scenario, name='create_scenario'),
    path('persona=<int:persona_id>/scenario=<int:scenario_id>/read_scenario', views.read_scenario, name='read_scenario'),
    path('persona=<int:persona_id>/scenario=<int:scenario_id>/update_scenario', views.update_scenario, name='update_scenario'),
    path('persona=<int:persona_id>/scenario=<int:scenario_id>/delete_scenario', views.delete_scenario, name='delete_scenario'),

    path('persona=<int:persona_id>/scenario=<int:scenario_id>/samples', views.samples, name='samples'),
    path('sample=<int:sample_id>', views.sample, name='sample'),

    path('failure_book', views.failure_book, name='failure_book'),
    path('read_ai', views.read_ai, name='read_ai')
]

