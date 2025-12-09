from django.urls import path
from .views import UploadView, QueryView, get_ollama_models, select_model

urlpatterns = [
    path("upload/", UploadView.as_view(), name="upload"),
    path("query/", QueryView.as_view(), name="query"),
    path("ollama/models/", get_ollama_models),
    path("ollama/set-model/", select_model),
]
