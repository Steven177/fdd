from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import ImageForm
from .models import Sample

def index(request):
  if request.method == 'POST':
    form = ImageForm(request.POST, request.FILES)

    if form.is_valid():
      form.save()
      obj = form.instance
      return render(request, 'fdd_app/index.html', {'obj': obj})

  else:
    form = ImageForm()
    samples = Sample.objects.all()

  return render(request, 'fdd_app/index.html', {'samples': samples, "form": form })
