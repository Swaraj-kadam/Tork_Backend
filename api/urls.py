from django.urls import path
from .views import UploadView, QueryView, get_models

urlpatterns = [
    path("upload/", UploadView.as_view(), name="upload"),
    path("query/", QueryView.as_view(), name="query"),
    path("models/", get_models, name="models"),
]
