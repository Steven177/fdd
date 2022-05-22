from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import ImageForm

"""
def index(request):
  # Process images uploaded by users
  if request.method == 'POST':
      form = ImageForm(request.POST, request.FILES)
      if form.is_valid():
          form.save()
          # Get the current instance object to display in the template
          img_obj = form.instance
          return render(request, 'fdd_app/index.html', {'form': form, 'img_obj': img_obj})
  else:
      form = ImageForm()
  return render(request, 'fdd_app/index.html', {'form': form})

"""

# Create your views here.
def index(request):

    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)

        if form.is_valid():
            form.save()
            img_obj = form.instance
            return render(request, 'fdd_app/index.html', {'form': form, 'img_obj': img_obj})
    else:
        form = ImageForm()
    return render(request, 'fdd_app/index.html', {'form' : form})
