

from django.urls import path
from .views import chat_function

urlpatterns = [
    
    path('chat/', chat_function, name='chat_function'),
]