from django import forms
from .models import Dataset

class DatasetUploadForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['name', 'file']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter dataset name'
            }),
            'file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.csv'
            })
        }

class DataCleaningForm(forms.Form):
    CLEANING_CHOICES = [
        ('remove_nulls', 'Remove null values'),
        ('fill_mean', 'Fill nulls with Mean'),
        ('fill_median', 'Fill nulls with Median'),
        ('fill_mode', 'Fill nulls with Mode'),
        ('drop_duplicates', 'Drop Duplicates'),
    ]
    
    cleaning_operation = forms.ChoiceField(
        choices=CLEANING_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
