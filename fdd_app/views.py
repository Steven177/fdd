import requests
import json
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from serpapi import GoogleSearch
import urllib.request
import torchvision.transforms as T
from io import BytesIO

from django.shortcuts import render, redirect, get_object_or_404, HttpResponseRedirect
from django.http import HttpResponse
from django.http import QueryDict
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import redirect
from django.db.models import Q


from .forms import PersonaForm, ScenarioForm, QueryForm
#from .forms import FailureForm

from .models import Sample, Expectation, Model_Prediction, Match, Ai, Persona, Scenario, Query, Suggestion

from .utils import *

from .ai import *
# from .seed import *

import replicate
import requests
import shutil
import urllib.request


# PERSONA
###################
def create_persona(request):
  persona_form = PersonaForm(request.POST, request.FILES)
  # POST
  if request.method == "POST":
    if persona_form.is_valid():
      persona_form.save()
      lastest_persona = Persona.objects.latest('id')
      return redirect('/fdd_app/persona={}/create_scenarios'.format(lastest_persona.id))

  elif Scenario.objects.all().exists():
    lastest_persona = Persona.objects.latest('id')
    latest_scenario = Scenario.objects.earliest('id')
    personas = Persona.objects.all().order_by("-id")
    return render(request, 'fdd_app/create_persona.html',
      {
      'latest_persona': lastest_persona,
      'latest_scenario': latest_scenario,
      'persona_form': persona_form,
      'personas': personas,
      })
  elif Persona.objects.all().exists():
    lastest_persona = Persona.objects.latest('id')
    personas = Persona.objects.all().order_by("-id")
    return render(request, 'fdd_app/create_persona.html',
      {
      'latest_persona': lastest_persona,
      'persona_form': persona_form,
      'personas': personas,
      })
  else:
    return render(request, 'fdd_app/create_persona.html',
      {
      'persona_form': persona_form,
      })


def update_persona(request, persona_id):
  lastest_persona = Persona.objects.latest('id')
  latest_scenario = Scenario.objects.earliest('id')

  persona = get_object_or_404(Persona, id = persona_id)
  personas = Persona.objects.all().order_by('-id')

  scenario = Scenario.objects.earliest('id')

  persona_form = PersonaForm(request.POST or None, instance = persona)
  scenario_form = ScenarioForm(request.POST)

  # POST
  if persona_form.is_valid():
    persona_form.save()
    return redirect('/fdd_app/persona={}/create_scenarios'.format(persona_id))

  # GET
  return render(request, "fdd_app/update_persona.html",
    {
    'latest_persona': lastest_persona,
    'latest_scenario': latest_scenario,
    'persona_form': persona_form,
    'scenario_form': scenario_form,
    'personas': personas,
    'persona':persona,
    'scenario': scenario
    })

def delete_persona(request, persona_id):
  persona = get_object_or_404(Persona, id = persona_id)
  persona.delete()

  if Persona.objects.all().count() > 0:
    persona = Persona.objects.latest('id')
    scenario = Scenario.objects.filter(persona=persona).first()
    return redirect('/fdd_app/persona={}/create_scenarios'.format(persona.id, scenario.id))
  else:
    return redirect('/fdd_app/create_persona')

def file_upload(request, persona_id):

  if request.method == 'POST':
    persona = Persona.objects.get(id=persona_id)
    my_file=request.FILES.get('file')
    scenario_id = request.POST.get("scenario_id")
    scenario = Scenario.objects.get(id=scenario_id)

    Sample.objects.create(persona=persona, scenario=scenario, image=my_file, uploaded=True)

    # Sample.objects.latest('id').delete()
    return redirect('/fdd_app/persona={}/create_scenarios'.format(persona_id, scenario_id))


#SCENARIO
###################
def create_scenarios(request, persona_id):
  lastest_persona = Persona.objects.latest('id')
  scenario_form = ScenarioForm(request.POST)
  personas = Persona.objects.all().order_by("-id")
  persona = Persona.objects.get(id=persona_id)

  if request.method == "POST":
    # scenario
    if scenario_form.is_valid():
      scenario_form = scenario_form.save(commit=False)
      scenario_form.persona = persona
      scenario_form.save()
      return redirect('/fdd_app/persona={}/create_scenarios'.format(persona_id))

  elif Scenario.objects.all().count() > 0:
    latest_scenario = Scenario.objects.latest('id')
    scenarios = Scenario.objects.filter(persona=persona).order_by("-id")
    scenario = Scenario.objects.earliest('id')
    samples_per_scenario = []
    for s in scenarios:
      samples = Sample.objects.filter(scenario=s, uploaded=True).order_by("-id")
      samples_per_scenario.append({'scenario': s, 'samples': samples})
    return render(request, 'fdd_app/create_scenarios.html',
      {
      'latest_persona': lastest_persona,
      'latest_scenario': latest_scenario,
      'scenario_form': scenario_form,
      'personas': personas,
      'persona': persona,
      'scenarios': scenarios,
      'scenario': scenario,
      'samples_per_scenario': samples_per_scenario,
      })

  else:
    return render(request, 'fdd_app/create_scenarios.html',
      {
      'latest_persona': lastest_persona,
      'scenario_form': scenario_form,
      'personas': personas,
      'persona': persona,
      })

def update_scenario(request, persona_id, scenario_id):
  lastest_persona = Persona.objects.latest('id')
  latest_scenario = Scenario.objects.earliest('id')

  scenario = get_object_or_404(Scenario, id = scenario_id)
  persona = Persona.objects.get(id=persona_id)
  personas = Persona.objects.all().order_by("-id")
  scenarios = Scenario.objects.filter(persona=persona).order_by("-id")

  top_scenario = Scenario.objects.earliest('id')

  samples_per_scenario = []
  for s in scenarios:
    samples = Sample.objects.filter(scenario=s).order_by("-id")
    samples_per_scenario.append({'scenario': s, 'samples': samples})

  scenario_form = ScenarioForm(request.POST or None, instance = scenario)
  # POST
  if scenario_form.is_valid():
    scenario_form.save()
    return redirect('/fdd_app/persona={}/create_scenarios'.format(persona_id, scenario_id))
  # GET
  return render(request, "fdd_app/update_scenario.html",
    {
    'latest_persona': lastest_persona,
    'latest_scenario': latest_scenario,
    'scenario_form': scenario_form,
    'scenario':scenario,
    'top_scenario': top_scenario,
    'scenarios': scenarios,
    'personas': personas,
    'persona': persona,
    'samples_per_scenario': samples_per_scenario
    })

def delete_scenario(request, persona_id, scenario_id):
  persona = Persona.objects.get(id=persona_id)
  scenario = get_object_or_404(Scenario, id = scenario_id)
  scenario.delete()
  return redirect('/fdd_app/persona={}/create_scenarios'.format(persona_id))


# SAMPLES
###################
@csrf_exempt
def samples(request, persona_id, scenario_id):

  latest_scenario = Scenario.objects.latest('id')
  lastest_persona = latest_scenario.persona

  query_form = QueryForm(request.POST)

  persona = Persona.objects.get(id=persona_id)
  scenario = Scenario.objects.get(id=scenario_id)

  personas = Persona.objects.all().order_by("-id")
  scenarios = Scenario.objects.all().order_by("-id")
  p_and_s = []

  for p in personas:
    s = p.scenario_set.all().order_by('-id')
    p_and_s.append({'persona': p, 'scenarios': s})

  samples = Sample.objects.filter(scenario=scenario).order_by("-id")

  ais = Ai.objects.all()


  # POST automatic
  if request.method == "POST" and query_form.is_valid():
    latest_sample = Sample.objects.latest('id')
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
      latest_sample = Sample.objects.latest("id")
      urllib.request.urlretrieve(url, 'media/images/google{}.jpg'.format(latest_sample.id))
      image1 = Sample.objects.create(image='../media/images/google{}.jpg'.format(latest_sample.id), persona=persona, scenario=scenario, generated=True)
      image1.save()

    # DALLE
    # client = replicate.Client(api_token=os.environ['REPLICATE_API_TOKEN'])

    model = replicate.models.get("stability-ai/stable-diffusion")
    generated_image = model.predict(prompt=query.input_query)

    # min_dalle = replicate.models.get("kuprel/min-dalle")
    # generated_image = min_dalle.predict(text=query.input_query, grid_size=1, temperature=1, progressive_outputs=False)

    #img = Image.open(output_path)
    #img = img.resize((256, 256))
    #img.save("resized.png")

    for url in generated_image:
      latest_sample = Sample.objects.latest('id')

      # img = Image.open(urlopen(url))
      # img = Image.open(requests.get(url, stream=True).raw)
      #response = requests.get(url)
      # img = Image.open(BytesIO(response.content))

      #response = requests.get(url)
      #img = Image.open(url)

      response = requests.get(url)
      image_bytes = io.BytesIO(response.content)
      img = Image.open(image_bytes)

      img = img.resize((256, 256))
      img.save("media/images/nn{}.jpg".format(latest_sample.id + 1))

      image2 = Sample.objects.create(image='../media/images/nn{}.jpg'.format(latest_sample.id + 1), persona=persona, scenario=scenario, generated=True)
      image2.save()


    return redirect('/fdd_app/persona={}/scenario={}/samples'.format(persona_id, scenario_id))

  # GET samples
  else:
    return render(request, 'fdd_app/samples.html',
      {
      'latest_persona': lastest_persona,
      'latest_scenario': latest_scenario,
      'query_form':query_form,
      'samples': samples,
      'persona': persona,
      'scenario': scenario,
      'ais': ais,
      'p_and_s': p_and_s,
      })

# READ_SAMPLE
###################
@csrf_exempt
def read_sample(request, persona_id, scenario_id, sample_id):
  lastest_persona = Persona.objects.latest('id')
  latest_scenario = Scenario.objects.latest('id')

  ais = Ai.objects.all()
  query_form = QueryForm(request.POST)

  sample = Sample.objects.get(id=sample_id)
  persona = Persona.objects.get(id=persona_id)
  scenario = Scenario.objects.get(id=scenario_id)

  personas = Persona.objects.all().order_by("-id")
  scenarios = Scenario.objects.all().order_by("-id")
  p_and_s = []

  for p in personas:
    s = p.scenario_set.all().order_by("-id")
    p_and_s.append({'persona': p, 'scenarios': s})

  samples = Sample.objects.filter(scenario=scenario).order_by("-id")

  colors = ["green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua","green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua", "green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua", "green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua", "green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua","green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua", "green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua","green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua", "green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua","green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua"]

  model_predictions = Model_Prediction.objects.filter(sample=sample)
  matches = Match.objects.filter(sample=sample)
  expectations = Expectation.objects.filter(sample=sample)

  suggestions = Suggestion.objects.filter(sample=sample)

  # POST automatic
  if request.method == "POST" and query_form.is_valid():
    latest_sample = Sample.objects.latest('id')
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
      latest_sample = Sample.objects.latest("id")
      urllib.request.urlretrieve(url, 'media/images/google{}.jpg'.format(latest_sample.id))
      image1 = Sample.objects.create(image='../media/images/google{}.jpg'.format(latest_sample.id), persona=persona, scenario=scenario, generated=True)
      image1.save()

    # DALLE
    # client = replicate.Client(api_token=os.environ['REPLICATE_API_TOKEN'])

    model = replicate.models.get("stability-ai/stable-diffusion")
    generated_image = model.predict(prompt=query.input_query)

    # min_dalle = replicate.models.get("kuprel/min-dalle")
    # generated_image = min_dalle.predict(text=query.input_query, grid_size=1, temperature=1, progressive_outputs=False)

    #img = Image.open(output_path)
    #img = img.resize((256, 256))
    #img.save("resized.png")

    for url in generated_image:
      latest_sample = Sample.objects.latest('id')

      # img = Image.open(urlopen(url))
      # img = Image.open(requests.get(url, stream=True).raw)
      #response = requests.get(url)
      # img = Image.open(BytesIO(response.content))

      #response = requests.get(url)
      #img = Image.open(url)

      response = requests.get(url)
      image_bytes = io.BytesIO(response.content)
      img = Image.open(image_bytes)

      img = img.resize((256, 256))
      img.save("media/images/nn{}.jpg".format(latest_sample.id + 1))

      image2 = Sample.objects.create(image='../media/images/nn{}.jpg'.format(latest_sample.id + 1), persona=persona, scenario=scenario, generated=True)
      image2.save()

    return redirect('/fdd_app/persona={}/scenario={}/sample={}/read_sample'.format(persona_id, scenario_id, sample_id))

  # POST Failure form
  elif request.method == "POST" and 'failure_severity' in request.POST:
    response = request.POST
    failure_severities = response.getlist('failure_severity')
    # failure_effects = response.getlist('failure_effects')
    match_ids = response.getlist('match_id')

    # Save failure severity
    for idx, sev in enumerate(failure_severities):
      match_id = match_ids[idx]
      match = Match.objects.get(id=match_id)
      if sev != "":
        match.failure_severity = int(sev)
        match.save()

    """
    # Save failure effects
    for idx, eff in enumerate(failure_effects):
      match_id = match_ids[idx]
      match = Match.objects.get(id=match_id)
      if eff != "":
        match.failure_effects = eff
        match.save()
    """
    sample.assessed = True
    sample.save()

    # ---------------------------
    # SUGGESTION ENGINE

    for match in matches:
      # CHALLENGE
      if match.true_positive:
        random = np.random.randint(0,3)
        if random == 0:
          name = "A very small {}".format(match.expectation.label)
        elif random == 1:
          name = "A {} at night".format(match.expectation.label.capitalize())
        elif random == 2:
          name = "A black and white of a {}".format(match.expectation.label)
        else:
          name = "Many {}s".format(match.expectation.label)
        Suggestion.objects.create(sample=sample, match=match, name=name, challenge=True).save()

      # REPEAT
      elif match.false_detection and match.indistribution or match.missing_detection and match.indistribution:
        """
        print(" in here ######################")
        print(os.getcwd())
        print(match.sample.image.url)
        path = os.getcwd() + match.sample.image.url
        # BLIB
        image_captioner = replicate.models.get("salesforce/blip")
        output = image_captioner.predict(image="img.jpeg")

        #clip_prefix = replicate.models.get("rmokady/clip_prefix_caption")
        #output = clip_caption.predict(image="")
        #clip_caption = replicate.models.get("j-min/clip-caption-reward")
        #output = clip_prefix.predict(image="...")

        with open(filename, "rb") as f:
          data = f.read()

        print(output)

        for o in output:
          print(o)
          print(o["text"])
          print(o["text"].split(": ")[1])

        # crop
        left = 10
        top = 10
        right = 200
        bottom = 200
        pil_image_obj = Image.open("img.jpeg")
        im = pil_image_obj.crop([left, top, right, bottom])
        im.show()

        """
      # GUIDE
      elif match.outofdistribution:
        # These code snippets use an open-source library. http://unirest.io/python
        url = "https://wordsapiv1.p.rapidapi.com/words/{}".format(match.expectation.label)

        headers = {
          "X-RapidAPI-Key": os.environ.get('WORDS_API'),
          "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
        }
        response = requests.request("GET", url, headers=headers)

        response = json.loads(response.text)
        if "results" in response:
          for result in response["results"]:
            if "synonyms" in result:
              synonyms = result["synonyms"]
              for s in synonyms:
                if check_if_indistribution(s):
                  n = "{}".format(s)
                  sugg = Suggestion.objects.create(sample=sample, match=match, name=n, guide=True, expectation_label=match.expectation.label)
                  sugg.save()
            if "typeOf" in result:
              higher_level_objects = result["typeOf"]
              for h in higher_level_objects:
                if check_if_indistribution(h):
                  n = "{}".format(h)
                  sugg = Suggestion.objects.create(sample=sample, match=match, name=n, guide=True, expectation_label=match.expectation.label)
                  sugg.save()

            if "hasTypes" in result:
              lower_level_objects = result["hasTypes"]
              for l in lower_level_objects:
                if check_if_indistribution(l):
                  n = "{}".format(l)
                  sugg = Suggestion.objects.create(sample=sample, match=match, name=n, guide=True, expectation_label=match.expectation.label)
                  sugg.save()
            break

    return redirect("/fdd_app/persona={}/scenario={}/sample={}/read_sample".format(persona_id, scenario_id, sample_id))

  # POST user (submit expectation)
  elif request.method == 'POST' and "expectation_submit" in request.POST:
    # ---------------------------
    # SAVE LABELS of EXPECTATION
    labels = request.POST.getlist("label")
    xs = request.POST.getlist("x")
    ys = request.POST.getlist("y")
    widths = request.POST.getlist("width")
    heights = request.POST.getlist("height")

    for idx, label in enumerate(labels):
      if label != "":
        if check_if_indistribution(label):
          indist = True
          outdist = False
        else:
          outdist = True
          indist = False

        new_exp = Expectation(
          sample=sample,
          label=label,
          indist=indist,
          outdist=outdist,
          xmin=int(xs[idx]),
          ymin=int(ys[idx]),
          xmax=int(widths[idx])+int(xs[idx]),
          ymax=int(heights[idx])+int(ys[idx]),
          )
        new_exp.save()

    sample.labelled = True
    sample.save()

    expectations = Expectation.objects.filter(sample=sample.id)

    # ---------------------------
    # MODEL PREDICTION 1
    if not sample.tested:
      model_prediction = make_prediction(sample.image.path) # AI model prediction
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
        print("exp_ind {}".format(exp_ind))
        print("pred_ind {}".format(pred_ind))
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
          if check_if_indistribution(match.expectation.label):
            match.indistribution = True
          else:
            match.outofdistribution = True
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
          if check_if_indistribution(match.expectation.label):
            match.indistribution = True
          else:
            match.outofdistribution = True
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

        sample.tested = True
        sample.save()
    return redirect('/fdd_app/persona={}/scenario={}/sample={}/read_sample'.format(persona_id, scenario_id, sample_id))


    """
    if request.POST['done_or_continue'][0] == "D":
      return redirect('/fdd_app/failure_book')
    elif request.POST['done_or_continue'][0] =="C":
      return redirect('/fdd_app/persona={}/scenario={}/samples'.format(persona_id, scenario_id))
    """
  # GET sample
  else:
    return render(request, 'fdd_app/read_sample.html', {
      'suggestions': suggestions,
      'latest_persona': lastest_persona,
      'latest_scenario': latest_scenario,
      'p_and_s': p_and_s,
      'query_form': query_form,
      'samples': samples,
      'persona': persona,
      'scenario': scenario,
      'sample': sample,
      'colors': colors,
      'personas': personas,
      'ais': ais,
      'model_predictions': model_predictions,
      'matches': matches,
      'expectations': expectations
    })

def update_sample(request, persona_id, scenario_id, sample_id):
  lastest_persona = Persona.objects.latest('id')
  latest_scenario = Scenario.objects.earliest('id')

  persona = Persona.objects.get(id=persona_id)
  scenario = Scenario.objects.get(id=scenario_id)
  sample = Sample.objects.get(id=sample_id)
  return render(request, 'fdd_app/update_sample.html',
    {
    'latest_persona': lastest_persona,
    'latest_scenario': latest_scenario,
    'persona':persona,
    'scenario': scenario,
    'sample': sample
    })

def delete_sample(request, persona_id, scenario_id, sample_id):

  sample = Sample.objects.get(id=sample_id)
  if sample.uploaded:
    sample.delete()
    return redirect('/fdd_app/persona={}/create_scenarios'.format(persona_id))
  else:
    sample.delete()
    return redirect('/fdd_app/persona={}/scenario={}/samples'.format(persona_id, scenario_id))


# FAILURE EXPLORATION
###################
def failure_exploration(request, persona_id, scenario_id, sample_id):
  sample = Sample.objects.get(id=sample_id)
  expectations = Expectation.objects.filter(sample=sample)

  lastest_persona = Persona.objects.latest('id')
  latest_scenario = Scenario.objects.earliest('id')

  ais = Ai.objects.all()
  query_form = QueryForm(request.POST)

  persona = Persona.objects.get(id=persona_id)
  scenario = Scenario.objects.get(id=scenario_id)

  personas = Persona.objects.all().order_by("-id")
  scenarios = Scenario.objects.all().order_by("-id")
  p_and_s = []

  for p in personas:
    s = p.scenario_set.all().order_by("-id")
    p_and_s.append({'persona': p, 'scenarios': s})

  samples = Sample.objects.filter(scenario=scenario).order_by("-id")

  colors = ["green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua","green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua", "green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua", "green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua", "green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua","green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua", "green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua","green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua", "green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua","green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua"]

  if sample.expectation_set.count() == 0:
    write_expectation = True
  else:
    write_expectation = False

  # POST Failure form
  if request.method == "POST" and 'failure_severity' in request.POST:
    response = request.POST
    failure_severities = response.getlist('failure_severity')
    failure_effects = response.getlist('failure_effects')
    match_ids = response.getlist('match_id')

    # Save failure severity
    for idx, sev in enumerate(failure_severities):
      match_id = match_ids[idx]
      match = Match.objects.get(id=match_id)
      if sev != "":
        match.failure_severity = int(sev)
        match.save()

    # Save failure effects
    for idx, eff in enumerate(failure_effects):
      match_id = match_ids[idx]
      match = Match.objects.get(id=match_id)
      if eff != "":
        match.failure_effects = eff
        match.save()
    return redirect("/fdd_app/persona={}/scenario={}/samples".format(persona_id, scenario_id))

  # Get
  else:
    print("Running GET failure exploration ...")
    print(sample.image.path)
    # ---------------------------
    # MODEL PREDICTION
    model_prediction = make_prediction(sample.image.path) # AI model prediction
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
      print("l1_cost_mat = {}".format(l1_cost_mat))
      giou_cost_mat = generalized_box_iou_loss(expectations, model_predictions)
      print("giou_cost_mat = {}".format(giou_cost_mat))
      loss_box = calculate_box_loss(giou_cost_mat, l1_cost_mat)
      print("loss_box = {}".format(loss_box))

      # labels_exp, labels_pred = padd_labels(labels_exp, labels_pred)
      loss_labels = calculate_class_loss(expectations, model_predictions)
      print("loss_labels = {}".format(loss_labels))

      loss_matching = calculate_matching_loss(loss_box, loss_labels)
      print("loss_matching = {}".format(loss_matching))
      # Hungarian algorithm
      exp_ind, pred_ind = linear_sum_assignment(loss_matching)
      print("exp_ind = {}".format(exp_ind))
      print("pred_ind = {}".format(pred_ind))
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



    return render(request, 'fdd_app/failure_exploration.html', {
      'write_expectation': write_expectation,
      'latest_persona': lastest_persona,
      'latest_scenario': latest_scenario,
      'model_predictions': model_predictions,
      'p_and_s': p_and_s,
      'query_form': query_form,
      'samples': samples,
      'persona': persona,
      'scenario': scenario,
      'sample': sample,
      'colors': colors,
      'personas': personas,
      'matches': matches,
      'ais': ais,
    })

# FAILURE BOOK
###################
@csrf_exempt
def failure_analysis(request, persona_id):
  # [sample: {'expectation', 'model_prediction'}]
  personas = Persona.objects.all().order_by("-id")
  persona = Persona.objects.get(id=persona_id)

  latest_scenario = Scenario.objects.latest('id')

  # ERRORS
  matches_per_persona = Match.objects.filter(sample__persona__id__exact = persona_id)
  len_matches = len(matches_per_persona)
  len_CD = Match.objects.filter(sample__persona__id__exact=persona_id, true_positive=True).count()
  len_FD = Match.objects.filter(sample__persona__id__exact=persona_id, false_detection=True).count()
  len_MD = Match.objects.filter(sample__persona__id__exact=persona_id, missing_detection=True).count()
  len_UD = Match.objects.filter(sample__persona__id__exact=persona_id, unnecessary_detection=True).count()

  # WARNINGS
  len_FTD= Match.objects.filter(sample__persona__id__exact=persona_id, failing_to_detect=True).count()
  len_CQB = Match.objects.filter(sample__persona__id__exact=persona_id, critical_quality_box=True).count()
  len_CQS = Match.objects.filter(sample__persona__id__exact=persona_id, critical_quality_score=True).count()
  len_warnings = len_FTD + len_CQB + len_CQS
  # INFOS
  len_ID = Match.objects.filter(sample__persona__id__exact=persona_id, indistribution=True).count()
  len_OOD = Match.objects.filter(sample__persona__id__exact=persona_id, outofdistribution=True).count()
  len_infos = len_ID + len_OOD

  print("############")
  print(matches_per_persona)
  print("############")
  num_sev = 0
  total_sev = 0

  for m in matches_per_persona:
    if m.failure_severity != 0:
     num_sev += 1
     total_sev += m.failure_severity

  if num_sev != 0:
    average_sev = total_sev / num_sev
  else:
    average_sev = 0

  card = {}
  for m in matches_per_persona:
    if m.expectation != None:
      exp_label = m.expectation.label
      exp_box = {"xmin": m.expectation.xmin, "ymin": m.expectation.ymin, "xmax": m.expectation.xmax, "ymax": m.expectation.ymax}
    else:
      exp_label = ""
      exp_box = ""

    if m.model_prediction != None:
      pred_label = m.model_prediction.label
      pred_box = {"xmin": m.model_prediction.xmin, "ymin": m.model_prediction.ymin, "xmax": m.model_prediction.xmax, "ymax": m.model_prediction.ymax}
    else:
      pred_label = ""
      pred_box = ""

    card[m.id] = {
    "exp_label": exp_label,
    "exp_box": exp_box,
    "pred_label":pred_label,
    "pred_box": pred_box,
    "tp": m.true_positive,
    "fd": m.false_detection,
    "md": m.missing_detection,
    "ud": m.unnecessary_detection,
    "ftd": m.failing_to_detect,
    "cqs": m.critical_quality_score,
    "cqb": m.critical_quality_box,
    "id": m.indistribution,
    "ood": m.outofdistribution,
    "sev": m.failure_severity
    }
  card = json.dumps(card, indent = 4)


  return render(request, 'fdd_app/failure_analysis.html',
    {
    'latest_scenario': latest_scenario,
    'personas': personas,
    'persona': persona,
    'matches_per_persona': matches_per_persona,
    'len_matches': len_matches,
    'len_warnings': len_warnings,
    'len_infos': len_infos,
    'len_CD': len_CD,
    'len_FD': len_FD,
    'len_MD': len_MD,
    'len_UD': len_UD,
    'len_FTD': len_FTD,
    'len_CQB': len_CQB,
    'len_CQS': len_CQS,
    'len_ID': len_ID,
    'len_OOD': len_OOD,
    'average_sev': average_sev,
    'card': card
    })



def read_ai(request):
  return render(request, 'fdd_app/read_ai.html')

def data(request):
  colors = ["green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua","green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua", "green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua", "green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua", "green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua","green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua", "green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua","green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua", "green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua","green", "blue", "red", "yellow", "purple", "fuchsia", "olive", "navy", "teal", "aqua"]
  finetuning = []
  retraining = []
  samples = Sample.objects.all().order_by("-id")
  for sample in samples:
    indist_exps = sample.expectation_set.filter(indist=True)
    exps = sample.expectation_set.all()
    model_preds = sample.model_prediction_set.all()
    finetuning.append({'sample': sample, 'expectations': indist_exps, 'model_predictions': model_preds})
    retraining.append({'sample': sample, 'expectations': exps, 'model_predictions': model_preds})
  return render(request, 'fdd_app/data.html',
    {
    'samples': samples,
    'finetuning': finetuning,
    'retraining': retraining,
    'colors': colors,
    })

def augmentations(request, persona_id, scenario_id, sample_id):
  print("OKAY")
  # AUGMENTATIONS
  persona = Persona.objects.get(id=persona_id)
  scenario = Scenario.objects.get(id=scenario_id)
  sample = Sample.objects.get(id=sample_id)
  pil_img = Image.open(sample.image)

  b = T.functional.adjust_brightness(pil_img, np.random.randint(2, 3))
  c = T.functional.adjust_contrast(pil_img, np.random.randint(3, 4))
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
  g_new.persona = persona
  g_new.scenario = scenario
  g_new.save()
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

  return redirect("/fdd_app/persona={}/scenario={}/sample={}/read_sample".format(persona_id, scenario_id, sample_id))

