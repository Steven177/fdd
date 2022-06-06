from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
from .forms import ImageForm
from .models import Sample
from .utils import *
import requests
import json

def index(request):
  if request.method == 'POST':
    form = ImageForm(request.POST, request.FILES)

    if form.is_valid():
      form.save()
      sample = form.instance
      model_prediction = query(sample.image.path)
      colors = ["green", "blue", "red", "yellow", "purple"]

      return render(request, 'fdd_app/index.html', {
        'form': form,
        'sample': sample,
        'model_prediction': model_prediction,
        'colors': colors,
        })

  else:
    form = ImageForm(request.POST, request.FILES)
    return render(request, 'fdd_app/index.html', {"form": form })

def false_observation(request):
  return render(request, 'fdd_app/false_observation.html')

def failing_to_observe(request):
  return render(request, 'fdd_app/failing_to_observe.html')
