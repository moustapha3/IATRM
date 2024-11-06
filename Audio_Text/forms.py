# forms.py dans votre application
from django import forms
from .models import AudioModel

class AudioModelForm(forms.ModelForm):
    class Meta:
        model = AudioModel
        fields = ['audioname', 'audio']
