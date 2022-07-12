from django import forms
from django.forms import formset_factory
from .models import Sample
from .models import Failure
from .models import Persona
# from .models import Expectation


class ImageForm(forms.ModelForm):
  class Meta:
      model = Sample
      fields = ['image']

class FailureForm(forms.ModelForm):
  class Meta:
    model = Failure
    fields = '__all__'
    exlude = ('sample',)

class PersonaForm(forms.ModelForm):
  class Meta:
    model = Persona
    fields = '__all__'

"""
class ExpectationForm(forms.ModelForm):
  class Meta:
    model = Expectation
    fields = '__all__'
"""
