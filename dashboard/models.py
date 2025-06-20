from django.db import models
import pandas as pd
import os
from django.conf import settings

class Dataset(models.Model):
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    original_shape = models.CharField(max_length=100, blank=True)
    cleaned_shape = models.CharField(max_length=100, blank=True)
    
    def __str__(self):
        return self.name
    
    def get_dataframe(self):
        """Load the dataset as a pandas DataFrame"""
        try:
            # Check if file exists
            if not self.file:
                print("No file attached to dataset")
                return None
                
            file_path = self.file.path
            print(f"Trying to load file from: {file_path}")
            
            if not os.path.exists(file_path):
                print(f"File does not exist at path: {file_path}")
                return None
            
            # Check file size
            file_size = os.path.getsize(file_path)
            print(f"File size: {file_size} bytes")
            
            if file_size == 0:
                print("File is empty")
                return None
            
            # Try to read the CSV with different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    print(f"Trying encoding: {encoding}")
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"Successfully loaded with {encoding}. Shape: {df.shape}")
                    return df
                except UnicodeDecodeError:
                    print(f"Failed with encoding: {encoding}")
                    continue
                except Exception as e:
                    print(f"Error with encoding {encoding}: {e}")
                    continue
            
            # If all encodings fail, try without specifying encoding
            try:
                df = pd.read_csv(file_path)
                print(f"Successfully loaded without encoding specification. Shape: {df.shape}")
                return df
            except Exception as e:
                print(f"Final attempt failed: {e}")
                return None
                
        except Exception as e:
            print(f"Error loading dataframe: {e}")
            return None
    
    def get_columns(self):
        """Get column names from the dataset"""
        df = self.get_dataframe()
        if df is not None:
            return list(df.columns)
        return []
    
    def get_null_counts(self):
        """Get null counts for each column"""
        df = self.get_dataframe()
        if df is not None:
            return df.isnull().sum().to_dict()
        return {}
