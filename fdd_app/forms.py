from django import forms
from django.forms import formset_factory
from .models import Sample
from .models import Persona
from .models import Scenario
from .models import Query
#from .models import Expectation


class ImageForm(forms.ModelForm):
  class Meta:
      model = Sample
      fields = ['image']

class PersonaForm(forms.ModelForm):
  class Meta:
    model = Persona
    fields = '__all__'

class ScenarioForm(forms.ModelForm):
  class Meta:
    model = Scenario
    fields = '__all__'


class QueryForm(forms.ModelForm):
  class Meta:
    model = Query
    fields = '__all__'

"""
class ExpectationForm(forms.ModelForm):
  class Meta:
    model = Expectation
    fields = '__all__'

class FailureForm(forms.ModelForm):
  class Meta:
    model = Failure
    fields = '__all__'
    exlude = ('sample',)
"""
