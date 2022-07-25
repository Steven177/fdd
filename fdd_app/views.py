import requests
import json
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from serpapi import GoogleSearch
import urllib.request
import torchvision.transforms as T

from django.shortcuts import render, redirect, get_object_or_404, HttpResponseRedirect
from django.http import HttpResponse
from django.http import QueryDict
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import redirect


from .forms import ImageForm, PersonaForm, ScenarioForm, QueryForm
#from .forms import FailureForm

from .models import Sample, Expectation, Model_Prediction, Match, Ai, Persona, Scenario, Query

from .utils import *
from .ai import *
from .dalle import *
from .creds import *
# from .seed import *

import replicate
import requests
import shutil

# PERSONA
###################
def create_persona(request):
  persona_form = PersonaForm(request.POST, request.FILES)
  ais = Ai.objects.all()
  # POST
  if request.method == "POST":
    if persona_form.is_valid():
      persona_form.save()
      last_persona = Persona.objects.latest('id')
      return redirect('/fdd_app/persona={}/create_scenario'.format(last_persona.id))
  # GET
  else:
    return render(request, 'fdd_app/create_persona.html',
      {
      'persona_form': persona_form,
      'ais': ais
      })

def read_persona(request, persona_id):
  persona = Persona.objects.get(id = persona_id)
  scenario = Scenario.objects.filter(persona=persona).first()
  return render(request, "fdd_app/read_persona.html", {'persona': persona, 'scenario': scenario})

def update_persona(request, persona_id):
  persona = get_object_or_404(Persona, id = persona_id)

  persona_form = PersonaForm(request.POST or None, instance = persona)
  # POST
  if persona_form.is_valid():
    persona_form.save()
    # return redirect('read_persona'.format(persona_id))
    # url = 'fdd_app/persona=' + str(persona_id) + '/read_persona'
    return redirect('read_persona', persona_id=persona_id)
  # GET
  return render(request, "fdd_app/update_persona.html", {'persona_form': persona_form, 'persona':persona})

def delete_persona(request, persona_id):
  persona = get_object_or_404(Persona, id = persona_id)
  persona.delete()

  persona = Persona.objects.earliest('id')
  scenario = Scenario.objects.filter(persona=persona).first()

  return redirect('/fdd_app/persona={}/scenario={}/samples'.format(persona.id, scenario.id))

#SCENARIO
###################

def create_scenario(request, persona_id):
  scenario_form = ScenarioForm(request.POST)
  persona = Persona.objects.get(id=persona_id)
  ais = Ai.objects.all()
  # POST
  if request.method == "POST":
    # Persona
    if scenario_form.is_valid():
      scenario_form = scenario_form.save(commit=False)
      scenario_form.persona = persona
      scenario_form.save()
      scenario_id = Scenario.objects.latest('id').id
      return redirect('/fdd_app/persona={}/scenario={}/samples'.format(persona_id, scenario_id))
  # GET
  else:
    return render(request, 'fdd_app/create_scenario.html',
      {
      'scenario_form': scenario_form,
      'persona': persona,
      'ais': ais
      })


def read_scenario(request, persona_id, scenario_id):
  persona = Persona.objects.get(id=persona_id)
  scenario = Scenario.objects.get(id=scenario_id)
  return render(request, "fdd_app/read_scenario.html", {'persona': persona, 'scenario': scenario})

def update_scenario(request, persona_id, scenario_id):
  scenario = get_object_or_404(Scenario, id = scenario_id)

  scenario_form = ScenarioForm(request.POST or None, instance = scenario)
  # POST
  if scenario_form.is_valid():
    scenario_form.save()

    return redirect('read_scenario', persona_id=persona_id, scenario_id=scenario_id)
  # GET
  return render(request, "fdd_app/update_scenario.html",
    {
    'scenario_form': scenario_form,
    'scenario':scenario
    })

def delete_scenario(request, persona_id, scenario_id):
  persona = Persona.objects.get(id=persona_id)
  scenario = get_object_or_404(Scenario, id = scenario_id)
  scenario.delete()
  scenario_id = Scenario.objects.latest('id').id

  return redirect('/fdd_app/persona={}/scenario={}/samples'.format(persona_id, scenario_id))

# SAMPLES
###################
@csrf_exempt
def samples(request, persona_id, scenario_id):
  image_form = ImageForm(request.POST, request.FILES)
  query_form = QueryForm(request.POST)

  persona = Persona.objects.get(id=persona_id)
  scenario = Scenario.objects.get(id=scenario_id)

  personas = Persona.objects.all()
  scenarios = Scenario.objects.all()
  p_and_s = []

  for p in personas:
    s = p.scenario_set.all()
    p_and_s.append({'persona': p, 'scenarios': s})

  samples = Sample.objects.filter(scenario=scenario)

  ais = Ai.objects.all()

  # POST manual
  if request.method == "POST" and image_form.is_valid():
    i_f = image_form.save(commit=False)
    i_f.persona = persona
    i_f.scenario = scenario
    i_f.save()

    # AUGMENTATIONS
    s = Sample.objects.latest('id')
    pil_img = Image.open(s.image)

    b = T.functional.adjust_brightness(pil_img, np.random.randint(2, 3))
    c = T.functional.adjust_contrast(pil_img, np.random.randint(1, 2))
    g = T.functional.gaussian_blur(pil_img, kernel_size=(31, 27), sigma=(100))
    r = T.functional.rotate(pil_img, np.random.randint(20, 360))
    h = T.functional.adjust_brightness(pil_img, np.random.uniform(0.2, 0.6))
    # b
    b_file = augment_image(b)
    b_new = Sample(image=b_file)
    b_new.persona = persona
    b_new.scenario = scenario
    b_new.save()
    # c
    c_file = augment_image(c)
    c_new = Sample(image=c_file)
    c_new.persona = persona
    c_new.scenario = scenario
    c_new.save()
    # g
    g_file = augment_image(g)
    g_new = Sample(image=g_file)
    c_new.persona = persona
    c_new.scenario = scenario
    c_new.save()
    # g
    r_file = augment_image(r)
    r_new = Sample(image=r_file)
    r_new.persona = persona
    r_new.scenario = scenario
    r_new.save()
    # g
    h_file = augment_image(h)
    h_new = Sample(image=h_file)
    h_new.persona = persona
    h_new.scenario = scenario
    h_new.save()

    return render(request, 'fdd_app/samples.html',
      {
      'image_form': image_form,
      'query_form':query_form,
      'samples': samples,
      'persona': persona,
      'scenario': scenario,
      'ais': ais,
      'p_and_s': p_and_s
      })

  # POST automatic
  elif request.method == "POST" and query_form.is_valid():
    # Google API
    qf = query_form.save(commit=False)
    qf.persona = persona
    qf.scenario = scenario
    qf.save()

    query = Query.objects.latest('id')
    results = call_google_api(query.input_query)

    image_results = results['images_results']

    random_idx = np.random.randint(0, np.floor(len(image_results)/4))
    image_results = image_results[random_idx:random_idx + 1]

    for image_result in image_results:
      url = image_result['thumbnail']
      title = image_result['title']
      urllib.request.urlretrieve(url, 'media/images/{}.jpg'.format(title))
      image1 = Sample.objects.create(image='../media/images/{}.jpg'.format(title), persona=persona, scenario=scenario)
      image1.save()

    # DALLE
    client = replicate.Client(api_token=REPLICATE_API_TOKEN)
    model = client.models.get("kuprel/min-dalle")
    generated_image = model.predict(text=query.input_query, grid_size=1, temperature=1, progressive_outputs=False)
    for url in generated_image:
      sample = Sample.objects.latest('id')
      file_name = 'media/images/{}.jpg'.format(sample.id + 1)
      res = requests.get(url, stream = True)

      if res.status_code == 200:
        with open(file_name,'wb') as f:
          shutil.copyfileobj(res.raw, f)
          image2 = Sample.objects.create(image='../media/images/{}.jpg'.format(sample.id + 1), persona=persona, scenario=scenario)
          image2.save()
        print('Image sucessfully Downloaded: ',file_name)
      else:
        print('Image Couldn\'t be retrieved')

    return render(request, 'fdd_app/samples.html',
      {
      'image_form': image_form,
      'query_form': query_form,
      'samples': samples,
      'persona': persona,
      'scenario': scenario,
      'ais': ais,
      'p_and_s': p_and_s
      })

  # GET samples
  else:
    print(p_and_s)
    return render(request, 'fdd_app/samples.html',
      {
      'image_form': image_form,
      'query_form':query_form,
      'samples': samples,
      'persona': persona,
      'scenario': scenario,
      'ais': ais,
      'p_and_s': p_and_s,
      })


@csrf_exempt
def sample(request, sample_id):
  personas = Persona.objects.all()
  ais = Ai.objects.all()
  # expectation_form = ExpectationForm(request.POST)
  # failure_form = FailureForm(request.POST)

  # template control
  write_expectation = True
  read_expectation = False

  sample = Sample.objects.get(id=sample_id)

  colors = ["green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua","green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua", "green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua", "green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua", "green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua","green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua", "green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua","green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua", "green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua","green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua"]

  # POST user (send boxes)
  if request.is_ajax():
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

    return render(request, 'fdd_app/sample.html', {
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
      giou_cost_mat = generalized_box_iou_loss(expectations, model_predictions)
      loss_box = calculate_box_loss(giou_cost_mat, l1_cost_mat)

      # labels_exp, labels_pred = padd_labels(labels_exp, labels_pred)
      loss_labels = calculate_class_loss(expectations, model_predictions)

      loss_matching = calculate_matching_loss(loss_box, loss_labels)
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

    return render(request, 'fdd_app/sample.html', {
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
  elif request.method == "POST" and 'failure_severity' in request.POST:
    response = request.POST
    failure_severities = response.getlist('failure_severity')
    match_ids = response.getlist('match_id')

    # Update failure severity
    for idx, sev in enumerate(failure_severities):
      match_id = match_ids[idx]
      match = Match.objects.get(id=match_id)
      if sev != "":
        match.failure_severity = int(sev)
        match.save()

    if request.POST['done_or_continue'][0] == "D":
      return redirect('/fdd_app/failure_book')
    elif request.POST['done_or_continue'][0] =="C":
      return redirect('/fdd_app/samples')

  # GET sample
  else:
    return render(request, 'fdd_app/sample.html', {
      'sample': sample,
      'colors': colors,
      'write_expectation': write_expectation,
      'read_expectation': read_expectation,
      'personas': personas,
      'ais': ais
    })

# FAILURE BOOK
###################
@csrf_exempt
def failure_book(request):
  # GET failure_book
  book = []
  personas = Persona.objects.all()
  for persona in personas:
    samples = Sample.objects.filter(has_failure=True, persona=persona)
    book.append({'persona': persona, 'samples': samples})

  print(book)

  ais = Ai.objects.all()

  false_detections = Match.objects.filter(false_detection=True)
  failing_to_detect = Match.objects.filter(failing_to_detect=True)
  missing_detections = Match.objects.filter(missing_detection=True)
  unnecessary_detections = Match.objects.filter(unnecessary_detection=True)
  critical_quality_boxes = Match.objects.filter(critical_quality_box=True)
  critical_quality_scores = Match.objects.filter(critical_quality_score=True)

  sev_false_detections = 0
  for i in false_detections:
    sev_false_detections += i.failure_severity

  sev_failing_to_detect = 0
  for i in failing_to_detect:
    sev_failing_to_detect += i.failure_severity

  sev_missing_detections = 0
  for i in missing_detections:
    sev_missing_detections += i.failure_severity

  sev_unnecessary_detections = 0
  for i in unnecessary_detections:
    sev_unnecessary_detections += i.failure_severity

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



    # print(rec_ind)

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
    'false_detections': false_detections,
    'failing_to_detect': failing_to_detect,
    'missing_detections': missing_detections,
    'unnecessary_detections': unnecessary_detections,
    'critical_quality_boxes': critical_quality_boxes,
    'critical_quality_scores': critical_quality_scores,
    'sev_false_detections': sev_false_detections,
    'sev_failing_to_detect': sev_failing_to_detect,
    'sev_missing_detections': sev_missing_detections,
    'sev_unnecessary_detections': sev_unnecessary_detections,
    "data": data,
    'personas': personas,
    'ais': ais
    })


