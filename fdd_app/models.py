from django.db import models
from .utils import *
from django.conf import settings

class Sample(models.Model):
  image = models.ImageField(upload_to='images/')
  # model_prediction =




