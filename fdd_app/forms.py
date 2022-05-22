from django import forms
from .models import Sample


class ImageForm(forms.ModelForm):
    """Form for the image model"""
    class Meta:
        model = Sample
        fields = ['image']
