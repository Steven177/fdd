from django.db import models

class Sample(models.Model):
  image = models.ImageField(upload_to='images/')
