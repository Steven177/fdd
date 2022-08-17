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

"""
class Failure(models.Model):
  # samples = models.ManyToManyField(Sample, blank=True)
  # sample = models.ForeignKey(Sample, on_delete=models.CASCADE, default=False, blank=True)
  # sample = models.ForeignKey(Sample, on_delete=models.CASCADE, default=-1, blank=True)
  sample_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

  # In- or Out-of-distribution error
  indistribution_error = models.BooleanField(default=False, blank=True)
  outofdistribution_error = models.BooleanField(default=False, blank=True)

  # failure mode
  false_observation = models.BooleanField(default=False, blank=True)
  incomplete_observation = models.BooleanField(default=False, blank=True)
  critical_quality = models.BooleanField(default=False, blank=True)
  violation = models.BooleanField(default=False,blank=True)
  context_error = models.BooleanField(default=False, blank=True)
  other_mode = models.CharField(default=False, max_length = 200, blank=True)

  # failure effects
  physical_harm = models.BooleanField(default=False, blank=True)
  psychological_harm = models.BooleanField(default=False, blank=True)
  financial_damage = models.BooleanField(default=False, blank=True)
  reputational_damage = models.BooleanField(default=False, blank=True)
  personal_privacy_violation = models.BooleanField(default=False, blank=True)
  annoyance = models.BooleanField(default=False, blank=True)
  discrimination = models.BooleanField(default=False, blank=True)
  exclusion = models.BooleanField(default=False, blank=True)
  harmful_content = models.BooleanField(default=False, blank=True)
  public_misinformation = models.BooleanField(default=False, blank=True)
  political_conflict = models.BooleanField(default=False, blank=True)
  legal_breech = models.BooleanField(default=False, blank=True)
  economic_damage = models.BooleanField(default=False, blank=True)
  public_privacy_breech = models.BooleanField(default=False, blank=True)

  # failure severity
  failure_severity = models.IntegerField(default=-1, validators=[MaxValueValidator(10),MinValueValidator(-1)], blank=True)

"""




