from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse
from .models import Dataset
import os
import pandas as pd

def debug_dataset(request, dataset_id):
    """Debug view to check dataset file"""
    if not request.user.is_superuser:
        return HttpResponse("Access denied", status=403)
    
    dataset = get_object_or_404(Dataset, id=dataset_id)
    
    debug_info = []
    debug_info.append(f"Dataset ID: {dataset.id}")
    debug_info.append(f"Dataset Name: {dataset.name}")
    debug_info.append(f"File Field: {dataset.file}")
    debug_info.append(f"File Name: {dataset.file.name}")
    
    try:
        debug_info.append(f"File Path: {dataset.file.path}")
        debug_info.append(f"File Exists: {os.path.exists(dataset.file.path)}")
        
        if os.path.exists(dataset.file.path):
            debug_info.append(f"File Size: {os.path.getsize(dataset.file.path)} bytes")
            
            # Try to read first few lines
            try:
                with open(dataset.file.path, 'r', encoding='utf-8') as f:
                    first_lines = [f.readline().strip() for _ in range(5)]
                debug_info.append("First 5 lines:")
                for i, line in enumerate(first_lines, 1):
                    debug_info.append(f"  Line {i}: {line[:100]}...")
            except Exception as e:
                debug_info.append(f"Error reading file: {e}")
            
            # Try pandas
            try:
                df = pd.read_csv(dataset.file.path)
                debug_info.append(f"Pandas Shape: {df.shape}")
                debug_info.append(f"Pandas Columns: {list(df.columns)}")
            except Exception as e:
                debug_info.append(f"Pandas Error: {e}")
        
    except Exception as e:
        debug_info.append(f"Error accessing file: {e}")
    
    return HttpResponse("<br>".join(debug_info))
