from django.contrib import messages
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

from jobs.models import JobPosting
from analytics_dashboard.models import SupportMessage
from prediction.querysets import get_accessible_assessments

from .forms import RegistrationForm, StyledAuthenticationForm
from .models import EmployerProfile, GraduateProfile, UserProfile


def register_view(request):
    form = RegistrationForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = form.save(commit=False)
        user.first_name = form.cleaned_data["first_name"]
        user.last_name = form.cleaned_data["last_name"]
        user.email = form.cleaned_data["email"]
        user.save()

        role = form.cleaned_data["role"]
        UserProfile.objects.create(
            user=user,
            role=role,
            phone_number=form.cleaned_data["phone_number"],
            location=form.cleaned_data["location"],
            bio=form.cleaned_data["bio"],
            admin_code_used=form.cleaned_data.get("admin_code", ""),
        )

        if role == "student":
            GraduateProfile.objects.create(
                user=user,
                degree_level=form.cleaned_data["degree_level"],
                field_of_study=form.cleaned_data["field_of_study"],
                graduation_year=form.cleaned_data["graduation_year"],
                experience_level=form.cleaned_data["experience_level"],
                skills_text=form.cleaned_data["skills_text"],
                career_goal=form.cleaned_data["career_goal"],
            )

        if role == "employer":
            EmployerProfile.objects.create(
                user=user,
                company_name=form.cleaned_data["company_name"],
                industry=form.cleaned_data["industry"],
                website=form.cleaned_data["website"],
                company_size=form.cleaned_data["company_size"],
                contact_person=form.cleaned_data["contact_person"],
            )
            JobPosting.objects.create(
                employer=user,
                title=form.cleaned_data["job_title"],
                industry=form.cleaned_data["industry"],
                location=form.cleaned_data["job_location"],
                description=form.cleaned_data["job_description"],
                required_skills=[item.strip() for item in form.cleaned_data["job_required_skills"].split(",") if item.strip()],
                minimum_score=form.cleaned_data.get("minimum_score") or 55,
            )

        login(request, user)
        messages.success(request, "Your account has been created successfully.")
        return redirect("prediction:dashboard")

    return render(request, "users/register.html", {"form": form})


def login_view(request):
    form = StyledAuthenticationForm(request, data=request.POST or None)
    if request.method == "POST" and form.is_valid():
        login(request, form.get_user())
        messages.success(request, "Welcome back.")
        return redirect("prediction:dashboard")
    return render(request, "users/login.html", {"form": form})


def logout_view(request):
    logout(request)
    messages.success(request, "You have been logged out.")
    return redirect("prediction:home")


@login_required
def account_overview(request):
    context = {
        "assessments": get_accessible_assessments(request.user)[:8],
        "employers": EmployerProfile.objects.select_related("user").order_by("company_name")[:8],
        "jobs": JobPosting.objects.filter(is_active=True).select_related("employer")[:8],
        "support_messages": SupportMessage.objects.filter(user=request.user).order_by("-created_at")[:8],
    }
    return render(request, "users/overview.html", context)
