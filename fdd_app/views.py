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
      obj = form.instance
      model_prediction = query(obj.image.path)
      sample = Sample.objects.all().last()
      return render(request, 'fdd_app/index.html', {
        'form': form,
        'obj': obj,
        'sample': sample,
        'model_prediction': model_prediction,
        })

  else:
    form = ImageForm(request.POST, request.FILES)
    return render(request, 'fdd_app/index.html', {"form": form })


def detail(request):
  return render(request, 'fdd_app/detail.svelte')
