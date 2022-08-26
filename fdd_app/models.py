from django.db import models
from .utils import *
from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
import uuid

class Persona(models.Model):
  image = models.ImageField(upload_to='personas/', default='personas/default_persona.jpeg')
  name = models.CharField(max_length=100, blank=True)
  description = models.TextField(blank=True)

class Scenario(models.Model):
  persona = models.ForeignKey(Persona, on_delete=models.CASCADE, blank=True)
  description = models.CharField(max_length=300, blank=True)

class Query(models.Model):
  persona = models.ForeignKey(Persona, on_delete=models.CASCADE, blank=True)
  scenario = models.ForeignKey(Scenario, on_delete=models.CASCADE, blank=True)
  input_query = models.CharField(max_length=200, blank=True)

class Ai(models.Model):
  name = models.CharField(max_length=100, blank=True)

class Sample(models.Model):
  persona = models.ForeignKey(Persona, on_delete=models.CASCADE, blank=True, default=-1)
  scenario = models.ForeignKey(Scenario, on_delete=models.CASCADE, blank=True, default=-1)
  image = models.ImageField(upload_to='images/')
  has_failure = models.BooleanField(default=False, blank=True)
  uploaded = models.BooleanField(default=False, blank=True)
  generated = models.BooleanField(default=False, blank=True)
  labelled = models.BooleanField(default=False, blank=True)
  tested = models.BooleanField(default=False, blank=True)
  assessed = models.BooleanField(default=False, blank=True)

class Model_Prediction(models.Model):
  sample = models.ForeignKey(Sample, on_delete=models.CASCADE, blank=True)
  label = models.CharField(max_length=100, blank=True)
  score = models.FloatField(default=-1,blank=True)
  xmin = models.IntegerField(default=-1, blank=True)
  ymin = models.IntegerField(default=-1, blank=True)
  xmax = models.IntegerField(default=-1, blank=True)
  ymax = models.IntegerField(default=-1, blank=True)

class Expectation(models.Model):
  sample = models.ForeignKey(Sample, on_delete=models.CASCADE, blank=True)
  label = models.CharField(max_length=100, blank=True)
  indist = models.BooleanField(default=False, blank=True)
  outdist = models.BooleanField(default=False, blank=True)
  xmin = models.IntegerField(default=-1, blank=True)
  ymin = models.IntegerField(default=-1, blank=True)
  xmax = models.IntegerField(default=-1, blank=True)
  ymax = models.IntegerField(default=-1, blank=True)

class Match(models.Model):
  sample = models.ForeignKey(Sample, on_delete=models.CASCADE, blank=True)
  expectation = models.ForeignKey(Expectation, on_delete=models.CASCADE, null=True, blank=True)
  exp_idx = models.IntegerField(default=-1, blank=True)
  model_prediction = models.ForeignKey(Model_Prediction, on_delete=models.CASCADE, null=True, blank=True)
  pred_idx = models.IntegerField(default=-1, blank=True)

  true_positive = models.BooleanField(default=False, blank=True)
  failing_to_detect = models.BooleanField(default=False, blank=True)
  false_detection = models.BooleanField(default=False, blank=True)
  indistribution = models.BooleanField(default=False, blank=True)
  outofdistribution = models.BooleanField(default=False, blank=True)
  missing_detection = models.BooleanField(default=False, blank=True)
  unnecessary_detection = models.BooleanField(default=False, blank=True)
  critical_quality_box = models.BooleanField(default=False, blank=True)
  critical_quality_score = models.BooleanField(default=False, blank=True)

  failure_severity = models.IntegerField(default=0, blank=True)
  failure_effects = models.CharField(max_length=200, blank=True)

class Suggestion(models.Model):
  sample = models.ForeignKey(Sample, on_delete=models.CASCADE, blank=True, default=None)
  match = models.ForeignKey(Match, on_delete=models.CASCADE, blank=True)
  name = models.CharField(max_length=500, blank=True)
  challenge = models.BooleanField(default=False, blank=True)
  guide = models.BooleanField(default=False, blank=True)
  challenge_again = models.BooleanField(default=False, blank=True)
  expectation_label = models.CharField(max_length=200, blank=True)





