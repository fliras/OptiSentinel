from django.urls import path

from .views import DescreverEventoView, DiagnosticarEventoView

urlpatterns = [
    path("descrever/", DescreverEventoView.as_view(), name="descrever"),
    path("diagnosticar/", DiagnosticarEventoView.as_view(), name="diagnosticar"),
]