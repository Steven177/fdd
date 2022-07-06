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
from .forms import FailureForm

from .models import Sample
from .models import Failure
from .models import Expectation
from .models import Model_Prediction
from .models import Match

from .utils import *
from .ai import *


@csrf_exempt
def input(request):
  image_form = ImageForm(request.POST, request.FILES)
  # POST input
  if request.method == "POST":
    if image_form.is_valid():
      image_form.save()
      return redirect('/fdd_app/user')

  # GET input
  else:
    return render(request, 'fdd_app/input.html', {"image_form": image_form})


@csrf_exempt
def user(request):
  # expectation_form = ExpectationForm(request.POST)
  failure_form = FailureForm(request.POST)

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
      'failure_form': failure_form,
      'colors': colors,
      'write_expectation': write_expectation,
      'read_expectation': read_expectation
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
    # model_prediction = query(sample.image.path) # AI model prediction
    # pil_image_obj = Image.open(sample.image)
    # model_prediction = predict(pil_image_obj)
    model_prediction = [{'score': 0.9308645725250244, 'label': 'person', 'box': {'xmin': 79, 'ymin': 13, 'xmax': 273, 'ymax': 183}}, {'score': 0.9893394112586975, 'label': 'tie', 'box': {'xmin': 132, 'ymin': 167, 'xmax': 162, 'ymax': 184}}]
    for obj in model_prediction:
      new_pred = Model_Prediction(sample=sample, label=obj['label'], score=obj['score'], xmin=obj['box']['xmin'], ymin=obj['box']['ymin'], xmax=obj['box']['xmax'], ymax=obj['box']['ymax'] )
      new_pred.save()

    model_predictions = Model_Prediction.objects.filter(sample=sample.id)

    # ---------------------------
    # COST

    l1_cost_mat = calculate_l1_loss(expectations, model_predictions, sample)
    giou_cost_mat= generalized_box_iou_loss(expectations, model_predictions)
    loss_box = calculate_box_loss(giou_cost_mat, l1_cost_mat)

    # labels_exp, labels_pred = padd_labels(labels_exp, labels_pred)
    loss_labels = calculate_class_loss(expectations, model_predictions)

    loss_matching = calculate_matching_loss(loss_box, loss_labels)

    # Hungarian algorithm
    exp_ind, pred_ind = linear_sum_assignment(loss_matching)
    matching_cost = loss_matching[exp_ind, pred_ind].sum()

    # ---------------------------
    # MATCHES
    for i, exp_idx in enumerate(exp_ind):
      exp_idx = np.int64(exp_idx).item()
      pred_idx = np.int64(pred_ind[i]).item()
      exp = expectations[exp_idx]
      pred = model_predictions[pred_idx]
      new_match = Match(sample=sample, expectation=exp, exp_idx=exp_idx, model_prediction=pred, pred_idx=pred_idx)
      new_match.save()

    for exp_idx, exp in enumerate(expectations):
      print(len(Match.objects.filter(expectation=exp.id)))
      if len(Match.objects.filter(expectation=exp.id)) == 0:
        new_match = Match(sample=sample, expectation=exp, exp_idx=exp_idx)
        new_match.save()

    for pred_idx, pred in enumerate(model_predictions):
      print(len(Match.objects.filter(model_prediction=pred.id)))
      if len(Match.objects.filter(model_prediction=pred.id)) == 0:
        new_match = Match(sample=sample, model_prediction=pred, pred_idx=pred_idx)
        new_match.save()

    print(" --------------------------------------")
    print(" --------------------------------------")

    print("l1_cost_mat")
    print(l1_cost_mat)

    print("giou_cost_mat")
    print(giou_cost_mat)

    print("loss_box")
    print(loss_box)

    print("loss_labels")
    print(loss_labels)

    print("loss_matching")
    print(loss_matching)

    print("exp_ind")
    print(exp_ind)

    print("pred_ind")
    print(pred_ind)

    # ---------------------------
    # ERROR ANALYSIS

    print(matches)
    for match in matches:
      if match.model_prediction == None:
        match.missing_detection = True
        match.indistribution = check_if_indistribution(match.expectation.label)
      elif match.expectation == None:
        match.additional_detection = True
        match.critical_quality_score = check_quality_of_score(match.model_prediction.score)
      elif match.expectation.label != match.model_prediction.label:
        match.false_observation = True
        match.indistribution = check_if_indistribution(match.expectation.label)
        match.critical_quality_box = check_quality_of_box(match.expectation, match.model_prediction)
        match.critical_quality_score = check_quality_of_score(match.model_prediction.score)
      else:
        match.true_positive = True
        match.critical_quality_box = check_quality_of_box(match.expectation, match.model_prediction)
        match.critical_quality_score = check_quality_of_score(match.model_prediction.score)


    print(" --------------------------------------")
    print(" --------------------------------------")
    matches = []

    return render(request, 'fdd_app/user.html', {
      'colors': colors,
      'sample': sample,
      'model_predictions': model_predictions,
      'failure_form': failure_form,
      'expectations': expectations,
      'write_expectation': write_expectation,
      'read_expectation': read_expectation,
      'matches': matches,
    })

  # POST Failure form
  elif request.method == "POST":
    print("FAILURE SUBMIT")
    print(failure_form.is_valid())
    failure_form.save(commit=False)
    sample = Sample.objects.latest('id')
    print("SAMPLE --------------------------------------")
    print(sample)
    print("FAILURE --------------------------------------")
    sample = Failure.objects.latest('id')
    print(failure)
    print("FAILURE SAMPLE BEFORE --------------------------------------")
    print(failure.sample)
    failure.sample = sample
    failure.save()
    print("FAILURE SAMPLE AFTER --------------------------------------")
    print(failure.sample)
    return redirect('/fdd_app/failure_book')

  # GET user
  else:
    sample = Sample.objects.latest('id')
    return render(request, 'fdd_app/user.html', {
      'sample': sample,
      'colors': colors,
      'failure_form': failure_form,
      'write_expectation': write_expectation,
      'read_expectation': read_expectation
    })

@csrf_exempt
def model(request):
  if request.method == "POST":
    pass
  else:
    return render(request, 'fdd_app/model.html')

@csrf_exempt
def failure(request):
  failure_form = FailureForm(request.POST)
  if request.method == "POST":
    if failure_form.is_valid():
      failure_form.save()
      sample = Sample.objects.latest('id')
      print("SAMPLE --------------------------------------")
      print(sample)
      failure = failure_form.instance
      print("FAILURE --------------------------------------")
      print(failure)
      print("FAILURE SAMPLE BEFORE --------------------------------------")
      print(failure.sample)
      failure.sample.add(sample)
      print("FAILURE SAMPLE AFTER --------------------------------------")
      print(failure.sample)
      return render(request, 'fdd_app/open_model_exploration.html', {
        'image_form': image_form,
        'failure_form': failure_form,
        'colors': colors,
      })
  return render(request, 'fdd_app/failure.html')

@csrf_exempt
def failure_book(request):
  samples = Sample.objects.filter(failure=True)
  failures = Failure.objects.all()
  print(samples)
  return render(request, 'fdd_app/failure_book.html', {'samples': samples, 'failures': failures})


