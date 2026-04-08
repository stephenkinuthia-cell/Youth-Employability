from django.urls import path

from .views import recommendation_list

app_name = "recommendations"

urlpatterns = [
    path("", recommendation_list, name="list"),
]
