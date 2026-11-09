from django.urls import path

from .views import DescribeEventView, ValidateEventInputView, DiagnoseEventView

urlpatterns = [
    path("describe/", DescribeEventView.as_view(), name="events-describe"),
    path("validate/", ValidateEventInputView.as_view(), name="events-validate"),
    path("diagnose/", DiagnoseEventView.as_view(), name="events-diagnose"),
]