import requests
import json
import numpy as np
from PIL import Image

from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
from django.http import QueryDict
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import redirect

from scipy.optimize import linear_sum_assignment

from .forms import ImageForm
# from .forms import FailureForm
from .forms import PersonaForm

from .models import Sample
from .models import Failure
from .models import Expectation
from .models import Model_Prediction
from .models import Match
from .models import Persona
from .models import Ai

from .utils import *
from .ai import *
# from .seed import *

def persona(request):
  ais = Ai.objects.all()
  persona_form = PersonaForm(request.POST)
  # POST persona
  if request.method == "POST":
    print("persona_form.is_valid() {}".format(persona_form.is_valid()))
    if persona_form.is_valid():
      persona_form.save()
      return redirect('/fdd_app')
  # GET persona
  return render(request, 'fdd_app/persona.html', {'persona_form': persona_form, 'ais': ais})

@csrf_exempt
def input(request):
  image_form = ImageForm(request.POST, request.FILES)
  personas = Persona.objects.all()
  ais = Ai.objects.all()
  # POST input
  if request.method == "POST":
    if image_form.is_valid():
      image_form.save()
      return redirect('/fdd_app/user')

  # GET input
  else:
    return render(request, 'fdd_app/input.html', {"image_form": image_form, 'personas': personas, 'ais': ais})


@csrf_exempt
def user(request):
  personas = Persona.objects.all()
  ais = Ai.objects.all()
  # expectation_form = ExpectationForm(request.POST)
  # failure_form = FailureForm(request.POST)

  # template control
  write_expectation = True
  read_expectation = False

  colors = ["green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua","green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua", "green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua", "green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua", "green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua","green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua", "green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua","green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua", "green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua","green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua"]

  # POST user (send boxes)
  if request.is_ajax():

    sample = Sample.objects.latest('id')
    # expectation = expectation_form.instance

    exp_boxes = request.POST.get('expBoxes[]')

    exp_boxes = json.dumps(exp_boxes)
    exp_boxes = eval(json.loads(exp_boxes))

    for exp in exp_boxes:
      x = exp['x']
      y = exp['y']
      width = exp['width']
      height = exp['height']
      new_exp = Expectation(xmin=x, ymin=y, xmax=width + x, ymax=height + y, sample=sample)
      new_exp.save()

    return render(request, 'fdd_app/user.html', {
      'exp_boxes': exp_boxes,
      'sample': sample,
      'colors': colors,
      'write_expectation': write_expectation,
      'read_expectation': read_expectation,
      'personas': personas,
      'ais': ais
    })

  # POST user (submit expectation)
  elif request.method == 'POST' and "expectation_submit" in request.POST:
    # template control
    write_expectation = False
    read_expectation = True

    # ---------------------------
    # SAVE LABELS of EXPECTATION
    sample = Sample.objects.latest('id')
    expectations = Expectation.objects.filter(sample=sample.id)

    # https://stackoverflow.com/questions/42359112/access-form-data-of-post-method-in-django
    response = QueryDict.copy(request.POST)
    response.pop('csrfmiddlewaretoken')

    labels = []
    for key in response:
      if response[key] != "":
        labels.append(response[key])

    labels.pop(-1)

    for idx, label in enumerate(labels):
      exp = expectations[idx]
      exp.label = label
      exp.save()

    expectations = Expectation.objects.filter(sample=sample.id)

    # ---------------------------
    # MODEL PREDICTION
    model_prediction = query(sample.image.path) # AI model prediction
    # pil_image_obj = Image.open(sample.image)
    # model_prediction = predict(pil_image_obj)
    # model_prediction = [{'score': 0.9308645725250244, 'label': 'person', 'box': {'xmin': 79, 'ymin': 13, 'xmax': 273, 'ymax': 183}}, {'score': 0.9893394112586975, 'label': 'tie', 'box': {'xmin': 132, 'ymin': 167, 'xmax': 162, 'ymax': 184}}]
    for obj in model_prediction:
      new_pred = Model_Prediction(sample=sample, label=obj['label'], score=obj['score'], xmin=obj['box']['xmin'], ymin=obj['box']['ymin'], xmax=obj['box']['xmax'], ymax=obj['box']['ymax'] )
      new_pred.save()

    model_predictions = Model_Prediction.objects.filter(sample=sample.id)

    # ---------------------------
    # COST

    if len(expectations) > 0 and len(model_predictions) > 0:
      l1_cost_mat = calculate_l1_loss(expectations, model_predictions, sample)
      print("l1_cost_mat {}".format(l1_cost_mat))
      giou_cost_mat = generalized_box_iou_loss(expectations, model_predictions)
      print("giou_cost_mat {}".format(giou_cost_mat))
      loss_box = calculate_box_loss(giou_cost_mat, l1_cost_mat)
      print("loss_box {}".format(loss_box))

      # labels_exp, labels_pred = padd_labels(labels_exp, labels_pred)
      loss_labels = calculate_class_loss(expectations, model_predictions)
      print("loss_labels {}".format(loss_labels))

      loss_matching = calculate_matching_loss(loss_box, loss_labels)
      print("loss_matching {}".format(loss_matching))
      # Hungarian algorithm
      exp_ind, pred_ind = linear_sum_assignment(loss_matching)
      matching_cost = loss_matching[exp_ind, pred_ind].sum()

    else:
      exp_ind = []
      pred_ind = []
    # ---------------------------
    # MATCHES
    for i, exp_idx in enumerate(exp_ind):
      exp_idx = np.int64(exp_idx).item()
      pred_idx = np.int64(pred_ind[i]).item()
      exp = expectations[exp_idx]
      pred = model_predictions[pred_idx]

      # boxes need at least 0.2 IoU to be a match
      iou = calculate_iou(exp, pred)

      if iou >= 0.2:
        new_match = Match(sample=sample, expectation=exp, exp_idx=exp_idx, model_prediction=pred, pred_idx=pred_idx)
        new_match.save()

    for exp_idx, exp in enumerate(expectations):
      if len(Match.objects.filter(expectation=exp.id)) == 0:
        new_match = Match(sample=sample, expectation=exp, exp_idx=exp_idx)
        new_match.save()

    for pred_idx, pred in enumerate(model_predictions):
      if len(Match.objects.filter(model_prediction=pred.id)) == 0:
        new_match = Match(sample=sample, model_prediction=pred, pred_idx=pred_idx)
        new_match.save()

    print(" --------------------------------------")
    print(" --------------------------------------")

    print("exp_ind")
    print(exp_ind)

    print("pred_ind")
    print(pred_ind)

    print(" --------------------------------------")
    print(" --------------------------------------")

    # ---------------------------
    # ERROR ANALYSIS
    matches = Match.objects.filter(sample=sample.id)

    for match in matches:
      # Failing to detect
      if len(model_predictions) == 0:
        match.failing_to_detect = True
        match.save()
        sample.has_failure = True
        sample.save()


      # Missing detection
      if match.model_prediction == None:
        match.missing_detection = True
        match.indistribution = check_if_indistribution(match.expectation.label)
        match.save()
        sample.has_failure = True
        sample.save()

      # Unnecessary detection
      elif match.expectation == None:
        match.unnecessary_detection = True
        match.critical_quality_score = check_quality_of_score(match.model_prediction.score)
        match.save()
        sample.has_failure = True
        sample.save()

      # False detection
      elif match.expectation.label.lower() != match.model_prediction.label:
        match.false_detection = True
        match.indistribution = check_if_indistribution(match.expectation.label)
        match.critical_quality_box = check_quality_of_box(match.expectation, match.model_prediction)
        match.critical_quality_score = check_quality_of_score(match.model_prediction.score)
        match.save()
        sample.has_failure = True
        sample.save()

      # True positive
      else:
        match.true_positive = True
        match.critical_quality_box = check_quality_of_box(match.expectation, match.model_prediction)
        match.critical_quality_score = check_quality_of_score(match.model_prediction.score)
        match.save()

    return render(request, 'fdd_app/user.html', {
      'colors': colors,
      'sample': sample,
      'model_predictions': model_predictions,
      'expectations': expectations,
      'write_expectation': write_expectation,
      'read_expectation': read_expectation,
      'matches': matches,
      'personas': personas,
      'ais': ais
    })

  # POST Failure form
  elif request.method == "POST":
    response = request.POST
    failure_severities = response.getlist('failure_severity')
    match_ids = response.getlist('match_id')

    # Update failure severity
    for idx, sev in enumerate(failure_severities):
      match_id = match_ids[idx]
      match = Match.objects.get(id=match_id)
      match.failure_severity = int(sev)
      match.save()

    return redirect('/fdd_app/failure_book')

  # GET user
  else:
    sample = Sample.objects.latest('id')
    return render(request, 'fdd_app/user.html', {
      'sample': sample,
      'colors': colors,
      'write_expectation': write_expectation,
      'read_expectation': read_expectation,
      'personas': personas,
      'ais': ais
    })


@csrf_exempt
def failure(request):
  failure_form = FailureForm(request.POST)
  if request.method == "POST":
    if failure_form.is_valid():
      failure_form.save()
      sample = Sample.objects.latest('id')
      failure = failure_form.instance
      failure.sample.add(sample)
      return render(request, 'fdd_app/open_model_exploration.html', {
        'image_form': image_form,
        'failure_form': failure_form,
        'colors': colors,
      })
  return render(request, 'fdd_app/failure.html')

@csrf_exempt
def failure_book(request):
  # GET failure_book
  samples = Sample.objects.filter(has_failure=True)
  personas = Persona.objects.all()
  ais = Ai.objects.all()

  data = []
  for sample in samples:

    expectations = sample.expectation_set.all()
    model_predictions = sample.model_prediction_set.all()
    matches = sample.match_set.all()

    total_severity = 0
    num_of_errors = 0
    num_of_warn = 0

    recoveries = [
    "Quality of output",
    "N-best options",
    "Hand-over of control",
    "Implicit feedback",
    "Explicit feedback",
    "Corrections by the user",
    "Local explanation",
    "Global explanation"
    ]
    print(matches)
    rec_ind = []
    for match in matches:
      if match.failing_to_detect or match.false_detection or match.missing_detection or match.unnecessary_detection:
        num_of_errors += 1

      if match.critical_quality_score and match.critical_quality_box:
        num_of_warn += 2
      elif match.critical_quality_score or match.critical_quality_box:
        num_of_warn += 1

      total_severity += match.failure_severity

      # ---------------------------
      # FAILURE RECOVERY

      if match.failing_to_detect:
        rec_ind.append([2, 7, 3])
      if match.false_detection and match.indistribution:
        rec_ind.append([5, 6, 1])
      if match.false_detection and not match.indistribution:
        rec_ind.append([7, 4, 2])
      if match.missing_detection and match.indistribution:
        rec_ind.append([4, 5, 2])
      if match.missing_detection and  not match.indistribution:
        rec_ind.append([7, 4, 2])
      if match.unnecessary_detection:
        rec_ind.append([4])
      if match.critical_quality_score:
        rec_ind.append([0, 1, 2, 4])
      if match.critical_quality_box:
        rec_ind.append([0, 4, 5])

    print(" --------------------------------------")
    print(" --------------------------------------")

    print(rec_ind)

    print(" --------------------------------------")
    print(" --------------------------------------")

    data.append({
      "sample": sample,
      "matches": matches,
      "expectations": expectations,
      "model_predictions": model_predictions,
      "total_severity": total_severity,
      "num_of_errors": num_of_errors,
      "num_of_warn": num_of_warn,
      'personas': personas,
      'ais': ais
      })

  return render(request, 'fdd_app/failure_book.html',
    {
    'samples': samples,
    "data": data,
    "recoveries": recoveries,
    "rec_ind": rec_ind,
    'personas': personas,
    'ais': ais
    })


