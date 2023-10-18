

from django.urls import path
from .views import alishbafunction

urlpatterns = [
    
    path('alishba/', alishbafunction, name='alishbafunction'),
]