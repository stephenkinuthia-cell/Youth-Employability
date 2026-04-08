from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

from users.permissions import get_user_role

from .forms import JobPostingForm
from .models import JobPosting


@login_required
def job_list(request):
    role = get_user_role(request.user)
    form = JobPostingForm(request.POST or None)

    if request.method == "POST" and role in {"admin", "employer"} and form.is_valid():
        JobPosting.objects.create(
            employer=request.user,
            title=form.cleaned_data["title"],
            industry=form.cleaned_data["industry"],
            location=form.cleaned_data["location"],
            description=form.cleaned_data["description"],
            required_skills=[item.strip() for item in form.cleaned_data["required_skills"].split(",") if item.strip()],
            minimum_score=form.cleaned_data["minimum_score"],
        )
        messages.success(request, "Job listing added successfully.")
        return redirect("jobs:list")

    jobs = JobPosting.objects.filter(is_active=True).select_related("employer")
    my_jobs = jobs.filter(employer=request.user) if role in {"admin", "employer"} else JobPosting.objects.none()
    return render(request, "jobs/list.html", {"jobs": jobs, "form": form, "my_jobs": my_jobs, "role": role})
