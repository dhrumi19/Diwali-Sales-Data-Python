import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'data_analysis_dashboard.settings')
application = get_wsgi_application()
