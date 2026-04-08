from django.urls import path

from .views import account_overview, login_view, logout_view, register_view

app_name = "users"

urlpatterns = [
    path("", account_overview, name="overview"),
    path("register/", register_view, name="register"),
    path("login/", login_view, name="login"),
    path("logout/", logout_view, name="logout"),
]
