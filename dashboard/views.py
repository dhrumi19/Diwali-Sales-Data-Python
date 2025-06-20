from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error, mean_absolute_error
import io
import base64
import json
import os

from .models import Dataset
from .forms import DatasetUploadForm, DataCleaningForm

def home(request):
    """Home page with dataset upload"""
    if request.method == 'POST':
        form = DatasetUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # Check if file is actually a CSV
                uploaded_file = request.FILES['file']
                if not uploaded_file.name.lower().endswith('.csv'):
                    messages.error(request, 'Please upload a CSV file.')
                    return render(request, 'dashboard/home.html', {'form': form})
                
                # Check file size (limit to 50MB)
                if uploaded_file.size > 50 * 1024 * 1024:
                    messages.error(request, 'File size too large. Please upload a file smaller than 50MB.')
                    return render(request, 'dashboard/home.html', {'form': form})
                
                dataset = form.save()
                print(f"Dataset saved with ID: {dataset.id}")
                print(f"File path: {dataset.file.path}")
                
                # Try to load and validate the dataframe
                df = dataset.get_dataframe()
                if df is not None:
                    dataset.original_shape = f"{df.shape[0]} rows {df.shape[1]} columns"
                    dataset.save()
                    print(f"Dataset shape: {df.shape}")
                    print(f"Columns: {list(df.columns)}")
                    request.session['dataset_id'] = dataset.id
                    messages.success(request, f'Dataset uploaded successfully! Shape: {df.shape[0]} rows, {df.shape[1]} columns')
                    return redirect('data_options')
                else:
                    # Delete the dataset if we can't load it
                    dataset.delete()
                    messages.error(request, 'Could not read the CSV file. Please check the file format and try again.')
                    return render(request, 'dashboard/home.html', {'form': form})
                    
            except Exception as e:
                print(f"Error in home view: {e}")
                messages.error(request, f'Error uploading file: {str(e)}')
                return render(request, 'dashboard/home.html', {'form': form})
    else:
        form = DatasetUploadForm()
    
    return render(request, 'dashboard/home.html', {'form': form})

def data_options(request):
    """Main data options page"""
    dataset_id = request.session.get('dataset_id')
    if not dataset_id:
        messages.warning(request, 'No dataset found. Please upload a dataset first.')
        return redirect('home')
    
    try:
        dataset = get_object_or_404(Dataset, id=dataset_id)
        # Verify the dataset can still be loaded
        df = dataset.get_dataframe()
        if df is None:
            messages.error(request, 'Dataset file is no longer accessible. Please upload again.')
            return redirect('home')
        return render(request, 'dashboard/data_options.html', {'dataset': dataset})
    except Exception as e:
        print(f"Error in data_options: {e}")
        messages.error(request, 'Error accessing dataset. Please upload again.')
        return redirect('home')

def clean_data(request):
    """Data cleaning page"""
    dataset_id = request.session.get('dataset_id')
    if not dataset_id:
        messages.warning(request, 'No dataset found. Please upload a dataset first.')
        return redirect('home')
    
    try:
        dataset = get_object_or_404(Dataset, id=dataset_id)
        df = dataset.get_dataframe()
        
        # Handle case where dataframe couldn't be loaded
        if df is None:
            messages.error(request, 'Error loading dataset. The file may be corrupted or in an unsupported format.')
            return redirect('home')
        
        print(f"Loaded dataframe with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        if request.method == 'POST':
            form = DataCleaningForm(request.POST)
            if form.is_valid():
                operation = form.cleaned_data['cleaning_operation']
                
                try:
                    original_shape = df.shape
                    
                    # Apply cleaning operation
                    if operation == 'remove_nulls':
                        df = df.dropna()
                    elif operation == 'fill_mean':
                        numeric_columns = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_columns) > 0:
                            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
                        else:
                            messages.warning(request, 'No numeric columns found for mean filling.')
                    elif operation == 'fill_median':
                        numeric_columns = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_columns) > 0:
                            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
                        else:
                            messages.warning(request, 'No numeric columns found for median filling.')
                    elif operation == 'fill_mode':
                        for column in df.columns:
                            mode_val = df[column].mode()
                            if not mode_val.empty:
                                df[column] = df[column].fillna(mode_val[0])
                    elif operation == 'drop_duplicates':
                        df = df.drop_duplicates()
                    
                    # Update dataset info
                    dataset.cleaned_shape = f"{df.shape[0]} rows {df.shape[1]} columns"
                    dataset.save()
                    
                    # Store cleaned data in session
                    request.session['cleaned_data'] = df.to_json()
                    
                    messages.success(request, f'Data cleaned successfully! Shape changed from {original_shape} to {df.shape}. Applied: {operation}')
                    return redirect('clean_data')
                    
                except Exception as e:
                    print(f"Error during cleaning: {e}")
                    messages.error(request, f'Error during cleaning operation: {str(e)}')
                    return redirect('clean_data')
        else:
            form = DataCleaningForm()
        
        # Get null counts and data info
        try:
            null_counts = dataset.get_null_counts()
            # Limit preview to first 10 rows and handle large datasets
            preview_df = df.head(10)
            data_preview = preview_df.to_html(classes='table table-striped table-hover table-sm', 
                                            table_id='data-preview', 
                                            max_cols=10,
                                            max_rows=10)
        except Exception as e:
            print(f"Error creating preview: {e}")
            messages.error(request, f'Error processing data preview: {str(e)}')
            null_counts = {}
            data_preview = "<p>Unable to generate data preview</p>"
        
        context = {
            'dataset': dataset,
            'form': form,
            'null_counts': null_counts,
            'data_preview': data_preview,
            'original_shape': dataset.original_shape,
            'cleaned_shape': dataset.cleaned_shape or 'Not cleaned yet'
        }
        
        return render(request, 'dashboard/clean_data.html', context)
        
    except Exception as e:
        print(f"Error in clean_data view: {e}")
        messages.error(request, f'Error accessing dataset: {str(e)}')
        return redirect('home')

def visualize_data(request):
    """Data visualization page"""
    dataset_id = request.session.get('dataset_id')
    if not dataset_id:
        messages.warning(request, 'No dataset found. Please upload a dataset first.')
        return redirect('home')
    
    try:
        dataset = get_object_or_404(Dataset, id=dataset_id)
        
        # Get cleaned data if available, otherwise original
        cleaned_data = request.session.get('cleaned_data')
        if cleaned_data:
            try:
                df = pd.read_json(cleaned_data)
                print("Using cleaned data from session")
            except Exception as e:
                print(f"Error loading cleaned data: {e}")
                messages.warning(request, 'Error loading cleaned data. Using original dataset.')
                df = dataset.get_dataframe()
        else:
            df = dataset.get_dataframe()
            print("Using original dataset")
        
        # Handle case where dataframe couldn't be loaded
        if df is None:
            messages.error(request, 'Error loading dataset. Please upload a valid CSV file.')
            return redirect('home')
        
        if request.method == 'POST':
            plot_type = request.POST.get('plot_type')
            x_axis = request.POST.get('x_axis')
            y_axis = request.POST.get('y_axis')
            color = request.POST.get('color', 'blue')
            
            if not plot_type or not x_axis:
                messages.error(request, 'Please select plot type and X-axis column.')
                context = {
                    'dataset': dataset,
                    'columns': df.columns.tolist(),
                    'plot_generated': False
                }
                return render(request, 'dashboard/visualize_data.html', context)
            
            # Generate plot
            plt.figure(figsize=(12, 8))
            plt.style.use('default')
            
            try:
                if plot_type == 'bar':
                    if y_axis and y_axis in df.columns:
                        df.groupby(x_axis)[y_axis].mean().head(20).plot(kind='bar', color=color)
                        plt.ylabel(y_axis)
                    else:
                        df[x_axis].value_counts().head(20).plot(kind='bar', color=color)
                        plt.ylabel('Count')
                elif plot_type == 'line':
                    if y_axis and y_axis in df.columns:
                        plt.plot(df[x_axis], df[y_axis], color=color, alpha=0.7)
                        plt.ylabel(y_axis)
                    else:
                        plt.plot(df.index, df[x_axis], color=color, alpha=0.7)
                        plt.ylabel(x_axis)
                elif plot_type == 'scatter':
                    if y_axis and y_axis in df.columns:
                        plt.scatter(df[x_axis], df[y_axis], color=color, alpha=0.6)
                        plt.ylabel(y_axis)
                    else:
                        messages.error(request, 'Scatter plot requires both X and Y axis columns.')
                        raise ValueError("Y-axis required for scatter plot")
                elif plot_type == 'hist':
                    plt.hist(df[x_axis].dropna(), bins=30, color=color, alpha=0.7)
                    plt.ylabel('Frequency')
                elif plot_type == 'box':
                    df.boxplot(column=x_axis)
                    plt.ylabel(x_axis)
                elif plot_type == 'pie':
                    df[x_axis].value_counts().head(10).plot(kind='pie', autopct='%1.1f%%')
                elif plot_type == 'heatmap':
                    numeric_df = df.select_dtypes(include=[np.number])
                    if len(numeric_df.columns) > 1:
                        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
                    else:
                        plt.text(0.5, 0.5, 'Not enough numeric columns for heatmap\n(Need at least 2 numeric columns)', 
                                ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
                
                plt.title(f'{plot_type.title()} Chart - {x_axis}', fontsize=16, pad=20)
                plt.xlabel(x_axis)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save plot to base64 string
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                plot_data = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                context = {
                    'dataset': dataset,
                    'columns': df.columns.tolist(),
                    'plot_data': plot_data,
                    'plot_generated': True
                }
                
            except Exception as e:
                print(f"Error generating plot: {e}")
                messages.error(request, f'Error generating plot: {str(e)}')
                context = {
                    'dataset': dataset,
                    'columns': df.columns.tolist(),
                    'plot_generated': False
                }
        else:
            context = {
                'dataset': dataset,
                'columns': df.columns.tolist(),
                'plot_generated': False
            }
        
        return render(request, 'dashboard/visualize_data.html', context)
        
    except Exception as e:
        print(f"Error in visualize_data: {e}")
        messages.error(request, f'Error in visualization: {str(e)}')
        return redirect('home')

def ml_algorithms(request):
    """Machine learning algorithms selection"""
    dataset_id = request.session.get('dataset_id')
    if not dataset_id:
        messages.warning(request, 'No dataset found. Please upload a dataset first.')
        return redirect('home')
    
    dataset = get_object_or_404(Dataset, id=dataset_id)
    return render(request, 'dashboard/ml_algorithms.html', {'dataset': dataset})

def regression_models(request):
    """Regression models page"""
    dataset_id = request.session.get('dataset_id')
    if not dataset_id:
        messages.warning(request, 'No dataset found. Please upload a dataset first.')
        return redirect('home')
    return render(request, 'dashboard/regression_models.html')

def classification_models(request):
    """Classification models page"""
    dataset_id = request.session.get('dataset_id')
    if not dataset_id:
        messages.warning(request, 'No dataset found. Please upload a dataset first.')
        return redirect('home')
    return render(request, 'dashboard/classification_models.html')

def is_classification_target(y):
    """Determine if target variable is suitable for classification"""
    # Check if target is categorical/discrete
    if y.dtype == 'object' or y.dtype.name == 'category':
        return True
    
    # Check if numeric target has few unique values (likely categorical)
    unique_values = y.nunique()
    total_values = len(y)
    
    # If less than 10 unique values and they represent less than 5% of total data
    if unique_values <= 10 and unique_values / total_values < 0.05:
        return True
    
    # Check if all values are integers (might be encoded categories)
    if y.dtype in ['int64', 'int32'] and unique_values <= 20:
        return True
    
    return False

def train_model(request, model_type):
    """Train machine learning model"""
    dataset_id = request.session.get('dataset_id')
    if not dataset_id:
        messages.warning(request, 'No dataset found. Please upload a dataset first.')
        return redirect('home')
    
    try:
        dataset = get_object_or_404(Dataset, id=dataset_id)
        
        # Get cleaned data if available
        cleaned_data = request.session.get('cleaned_data')
        if cleaned_data:
            try:
                df = pd.read_json(cleaned_data)
            except Exception as e:
                messages.warning(request, 'Error loading cleaned data. Using original dataset.')
                df = dataset.get_dataframe()
        else:
            df = dataset.get_dataframe()
        
        # Handle case where dataframe couldn't be loaded
        if df is None:
            messages.error(request, 'Error loading dataset. Please upload a valid CSV file.')
            return redirect('home')
        
        # Determine if this is a regression or classification model
        is_regression_model = model_type in ['linear', 'polynomial', 'lasso', 'ridge', 'svm', 'random_forest']
        is_classification_model = model_type in ['logistic', 'decision_tree', 'knn']
        
        if request.method == 'POST':
            target_column = request.POST.get('target_column')
            feature_columns = request.POST.getlist('feature_columns')
            
            if not target_column or not feature_columns:
                messages.error(request, 'Please select target column and at least one feature column')
                context = {
                    'dataset': dataset,
                    'model_type': model_type,
                    'columns': df.columns.tolist(),
                    'model_trained': False
                }
                return render(request, 'dashboard/train_model.html', context)
            
            try:
                # Prepare data
                X = df[feature_columns].copy()
                y = df[target_column].copy()
                
                print(f"Original data shape - X: {X.shape}, y: {y.shape}")
                print(f"Target column '{target_column}' data type: {y.dtype}")
                print(f"Target unique values: {y.nunique()}")
                print(f"Sample target values: {y.head().tolist()}")
                
                # Check if target is appropriate for the model type
                target_is_classification = is_classification_target(y)
                
                if is_regression_model and target_is_classification:
                    messages.error(request, 
                        f'The target column "{target_column}" appears to be categorical. '
                        f'For regression models, please select a continuous numerical target (like Amount, Price, Quantity). '
                        f'Or use Classification models for categorical targets.')
                    context = {
                        'dataset': dataset,
                        'model_type': model_type,
                        'columns': df.columns.tolist(),
                        'model_trained': False
                    }
                    return render(request, 'dashboard/train_model.html', context)
                
                if is_classification_model and not target_is_classification:
                    # For classification with continuous target, we can bin it
                    messages.warning(request, 
                        f'Target column "{target_column}" is continuous. Converting to categories for classification.')
                    # Convert continuous target to categories (e.g., Low, Medium, High)
                    y = pd.cut(y, bins=3, labels=['Low', 'Medium', 'High'])
                
                # Handle categorical variables in features
                X = pd.get_dummies(X, drop_first=True)
                
                # Remove any remaining NaN values
                mask = ~(X.isnull().any(axis=1) | y.isnull())
                X = X[mask]
                y = y[mask]
                
                print(f"After cleaning - X: {X.shape}, y: {y.shape}")
                
                if len(X) < 10:
                    messages.error(request, 'Not enough valid data points for training (minimum 10 required)')
                    context = {
                        'dataset': dataset,
                        'model_type': model_type,
                        'columns': df.columns.tolist(),
                        'model_trained': False
                    }
                    return render(request, 'dashboard/train_model.html', context)
                
                # Split data
                test_size = min(0.3, max(0.1, len(X) * 0.2 / len(X)))  # Adaptive test size
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                
                # Select and train model
                if model_type == 'linear':
                    model = LinearRegression()
                elif model_type == 'polynomial':
                    model = Pipeline([
                        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                        ('linear', LinearRegression())
                    ])
                elif model_type == 'lasso':
                    model = Lasso(alpha=0.1, max_iter=1000)
                elif model_type == 'ridge':
                    model = Ridge(alpha=1.0)
                elif model_type == 'svm':
                    if is_regression_model:
                        model = SVR(kernel='rbf', C=1.0)
                    else:
                        model = SVC(kernel='rbf', C=1.0)
                elif model_type == 'random_forest':
                    if is_regression_model:
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                    else:
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                elif model_type == 'logistic':
                    model = LogisticRegression(random_state=42, max_iter=1000)
                elif model_type == 'decision_tree':
                    model = DecisionTreeClassifier(random_state=42)
                elif model_type == 'knn':
                    model = KNeighborsClassifier(n_neighbors=min(5, len(X_train)//2))
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics with proper formatting
                if is_regression_model:
                    # Regression metrics
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    
                    metrics = {
                        'MSE': f'{mse:.4f}',
                        'RMSE': f'{rmse:.4f}',
                        'R² Score': f'{r2:.4f}',
                        'MAE': f'{mae:.4f}'
                    }
                else:
                    # Classification metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    try:
                        f1 = f1_score(y_test, y_pred, average='weighted')
                    except:
                        f1 = 0.0
                    
                    metrics = {
                        'Accuracy': f'{accuracy:.4f}',
                        'F1 Score': f'{f1:.4f}'
                    }
                
                context = {
                    'dataset': dataset,
                    'model_type': model_type,
                    'columns': df.columns.tolist(),
                    'metrics': metrics,
                    'model_trained': True
                }
                
            except Exception as e:
                print(f"Error training model: {e}")
                error_msg = str(e)
                
                # Provide specific guidance for common errors
                if "continuous" in error_msg.lower() and "discrete" in error_msg.lower():
                    messages.error(request, 
                        'Target column contains continuous values but you selected a classification model. '
                        'Please either: 1) Choose a regression model for continuous targets, or '
                        '2) Select a categorical target column for classification.')
                else:
                    messages.error(request, f'Error training model: {error_msg}')
                
                context = {
                    'dataset': dataset,
                    'model_type': model_type,
                    'columns': df.columns.tolist(),
                    'model_trained': False
                }
        else:
            context = {
                'dataset': dataset,
                'model_type': model_type,
                'columns': df.columns.tolist(),
                'model_trained': False
            }
        
        return render(request, 'dashboard/train_model.html', context)
        
    except Exception as e:
        print(f"Error in train_model: {e}")
        messages.error(request, f'Error in model training: {str(e)}')
        return redirect('home')
