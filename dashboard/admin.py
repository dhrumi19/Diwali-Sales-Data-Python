from django.contrib import admin
from .models import Dataset

@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ['name', 'uploaded_at', 'original_shape', 'cleaned_shape']
    list_filter = ['uploaded_at']
    search_fields = ['name']
