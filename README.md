# 🔍 Data Analysis Dashboard
A comprehensive web-based data analysis platform built with Django, Bootstrap 5, and Python ML libraries. This project integrates data uploading, preprocessing, model training, and result visualization in one elegant interface.

## 🌟 Features

### 📊 Data Management
- **CSV File Upload** - Upload custom datasets with validation
- **Data Cleaning** - Remove nulls, fill missing values, drop duplicates
- **Data Preview** - Interactive tables showing cleaned data
- **Data Visualization** - Multiple chart types (Bar, Line, Scatter, Heatmap, etc.)

### 🤖 Machine Learning
- **Regression Models** - Linear, Polynomial, Lasso, Ridge, SVM
- **Classification Models** - Logistic Regression, Decision Tree, Random Forest, KNN
- **Model Training** - Custom feature and target selection
- **Performance Metrics** - Accuracy, F1 Score, R², RMSE, MAE

### 🎨 User Interface
- **Responsive Design** - Bootstrap 5 with custom styling
- **Interactive Charts** - Dynamic visualizations
- **Professional UI** - Clean cards and modern interface
- **Error Handling** - User-friendly error messages

## 🛠️ Tech Stack

### Backend
- **Django 4.2.7** - Web framework
- **Python 3.8+** - Programming language
- **SQLite** - Database (default)

### Frontend
- **HTML5** - Markup
- **Bootstrap 5.3.0** - CSS framework
- **JavaScript** - Interactivity
- **Font Awesome** - Icons

### Data Science & ML
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **matplotlib** - Static plotting
- **seaborn** - Statistical visualization
- **scikit-learn** - Machine learning

### 1. Clone the Repository
git clone https://github.com/yourusername/data-analysis-dashboard.git
cd data-analysis-dashboard

### 2. Create Virtual Environment
\`\`\`bash
# Windows
python -m venv venv
venv\Scripts\activate

### 3. Install Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 4. Setup Database
\`\`\`bash
python manage.py makemigrations
python manage.py migrate
\`\`\`

### 6. Run Development Server
\`\`\`bash
python manage.py runserver
\`\`\`

### 7. Access Application
Open your browser and go to: `http://127.0.0.1:8000/`

## 📁 Project Structure

\`\`\`
data-analysis-dashboard/
├── data_analysis_dashboard/    # Django project settings
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── dashboard/                  # Main Django app
│   ├── migrations/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── forms.py
│   ├── models.py
│   ├── urls.py
│   └── views.py
├── templates/                  # HTML templates
│   ├── base.html
│   └── dashboard/
│       ├── home.html
│       ├── data_options.html
│       ├── clean_data.html
│       ├── visualize_data.html
│       ├── ml_algorithms.html
│       ├── regression_models.html
│       ├── classification_models.html
│       └── train_model.html
├── media/                      # Uploaded files
│   └── datasets/
├── static/                     # Static files (CSS, JS, images)
├── requirements.txt            # Python dependencies
├── manage.py                   # Django management script

## 🎯 Usage Guide

### 1. Upload Dataset
- Navigate to the home page
- Upload a CSV file (max 50MB)
- Supported formats: CSV with headers

### 2. Clean Data
- Choose cleaning operations:
  - Remove null values
  - Fill nulls with mean/median/mode
  - Drop duplicates
- View data shape changes and null counts

### 3. Visualize Data
- Select plot type (Bar, Line, Scatter, Histogram, etc.)
- Choose X and Y axis columns
- Customize colors
- Generate interactive charts

### 4. Train ML Models

#### For Regression (Predicting Numbers):
- **Models**: Linear, Polynomial, Lasso, Ridge, SVM
- **Target**: Numerical columns (Amount, Price, Quantity)
- **Features**: Any relevant columns
- **Metrics**: MSE, RMSE, R² Score, MAE

#### For Classification (Predicting Categories):
- **Models**: Logistic Regression, Decision Tree, Random Forest, KNN
- **Target**: Categorical columns (Gender, Category, Status)
- **Features**: Any relevant columns
- **Metrics**: Accuracy, F1 Score

### 5. Example with Diwali Sales Data

**Regression Example:**
- Target: `Amount` (predict purchase amount)
- Features: `Age`, `Gender`, `Occupation`, `Product_Category`

**Classification Example:**
- Target: `Gender` (predict customer gender)
- Features: `Age`, `Amount`, `Occupation`, `Product_Category` 
