from django import forms

from .models import RawImage


class UploadImageForm(forms.ModelForm):
    class Meta:
        model = RawImage
        fields = ['image_file', ]
