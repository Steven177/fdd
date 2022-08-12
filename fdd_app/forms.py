from django import forms
from django.forms import formset_factory
from .models import Sample, Persona, Scenario, Query
"""
class ImageForm(forms.ModelForm):
  image = forms.ImageField(label="")
  class Meta:
      model = Sample
      fields = ['image']
"""
class PersonaForm(forms.ModelForm):
  image = forms.ImageField(label="", required=False)
  description = forms.CharField(widget=forms.Textarea(attrs={'onkeyup': 'showDescription();'}), label="", required=False)
  name = forms.CharField(widget=forms.TextInput(attrs={'onkeyup': 'showName();'}))

  class Meta:
    model = Persona
    fields = '__all__'

  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.fields['scenario'].queryset = Scenario.objects.none()
  """
class ScenarioForm(forms.ModelForm):
  description = forms.CharField(widget=forms.TextInput(attrs={'placeholder': 'Scenario'}), label="", required=False)
  class Meta:
    model = Scenario
    exclude = ['persona']

class QueryForm(forms.ModelForm):
  input_query = forms.CharField(widget=forms.TextInput(attrs={'placeholder': 'Prompt'}), label="", required=True)
  class Meta:
    model = Query
    fields = ['input_query']
