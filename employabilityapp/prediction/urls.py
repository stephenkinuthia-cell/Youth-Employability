from django.urls import path

from .views import assessment_detail, dashboard, download_report, home

app_name = "prediction"

urlpatterns = [
    path("", home, name="home"),
    path("predict/", dashboard, name="dashboard"),
    path("assessments/<int:pk>/", assessment_detail, name="assessment_detail"),
    path("assessments/<int:pk>/report/", download_report, name="download_report"),
]
