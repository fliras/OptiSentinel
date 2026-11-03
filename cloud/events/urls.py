from django.urls import path

from .views import DescribeEventView

urlpatterns = [
    path("describe/", DescribeEventView.as_view(), name="events-describe"),
]