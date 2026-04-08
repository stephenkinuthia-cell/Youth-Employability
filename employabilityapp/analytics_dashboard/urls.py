from django.urls import path

from .views import analytics_home, contact_admin, support_messages

app_name = "analytics_dashboard"

urlpatterns = [
    path("", analytics_home, name="home"),
    path("contact-admin/", contact_admin, name="contact_admin"),
    path("messages/", support_messages, name="messages"),
]
