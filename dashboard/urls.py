from django.urls import path
from . import views, debug_views

urlpatterns = [
    path('', views.home, name='home'),
    path('data-options/', views.data_options, name='data_options'),
    path('clean-data/', views.clean_data, name='clean_data'),
    path('visualize-data/', views.visualize_data, name='visualize_data'),
    path('ml-algorithms/', views.ml_algorithms, name='ml_algorithms'),
    path('regression-models/', views.regression_models, name='regression_models'),
    path('classification-models/', views.classification_models, name='classification_models'),
    path('train-model/<str:model_type>/', views.train_model, name='train_model'),
    path('debug-dataset/<int:dataset_id>/', debug_views.debug_dataset, name='debug_dataset'),
]
