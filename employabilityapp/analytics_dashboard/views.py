from collections import Counter
from datetime import timedelta

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from jobs.models import JobPosting
from prediction.models import EmployabilityAssessment
from recommendations.models import SkillRecommendation
from users.models import EmployerProfile
from users.permissions import get_user_role, role_required

from .forms import SupportMessageForm
from .models import AppVisit, SupportMessage

@login_required
@role_required("admin")
def analytics_home(request):
    if request.method == "POST":
        target_user = get_object_or_404(User, pk=request.POST.get("user_id"))
        action = request.POST.get("action")
        if action == "toggle_active":
            target_user.is_active = not target_user.is_active
            target_user.save(update_fields=["is_active"])
            messages.success(request, f"Updated account status for {target_user.username}.")
        elif action == "set_employer":
            target_user.platform_profile.role = "employer"
            target_user.platform_profile.save(update_fields=["role"])
            messages.success(request, f"{target_user.username} is now an employer.")
        elif action == "set_student":
            target_user.platform_profile.role = "student"
            target_user.platform_profile.save(update_fields=["role"])
            messages.success(request, f"{target_user.username} is now a user.")
        return redirect("analytics_dashboard:home")

    assessments = EmployabilityAssessment.objects.all()
    recommendations = SkillRecommendation.objects.all()
    jobs = JobPosting.objects.filter(is_active=True)
    visits = AppVisit.objects.all()

    skills_counter = Counter()
    for assessment in assessments:
        skills_counter.update(assessment.current_skills)

    gap_counter = Counter()
    for item in recommendations:
        gap_counter.update([item.title])

    last_7_days = timezone.now() - timedelta(days=7)
    context = {
        "assessment_count": assessments.count(),
        "low_risk_count": assessments.filter(risk_category="Low").count(),
        "high_risk_count": assessments.filter(risk_category="High").count(),
        "job_count": jobs.count(),
        "top_skills": skills_counter.most_common(6),
        "top_gaps": gap_counter.most_common(6),
        "recent_assessments": assessments[:6],
        "visits_total": visits.count(),
        "visits_last_week": visits.filter(created_at__gte=last_7_days).count(),
        "active_users": visits.filter(created_at__gte=last_7_days, user__isnull=False).values("user_id").distinct().count(),
        "avg_score": round(sum(assessments.values_list("employability_score", flat=True)) / assessments.count(), 2) if assessments.exists() else 0,
        "effectiveness_rate": round((assessments.filter(risk_category="Low").count() / assessments.count()) * 100, 1) if assessments.exists() else 0,
        "users": User.objects.select_related("platform_profile").order_by("-date_joined")[:20],
        "employers": EmployerProfile.objects.select_related("user").order_by("company_name")[:10],
    }
    return render(request, "analytics_dashboard/home.html", context)


@login_required
def contact_admin(request):
    role = get_user_role(request.user)
    if role == "admin":
        messages.info(request, "Admins can review issues from the admin inbox.")
        return redirect("analytics_dashboard:messages")

    initial = {"contact_email": request.user.email}
    form = SupportMessageForm(request.POST or None, initial=initial)
    if request.method == "POST" and form.is_valid():
        SupportMessage.objects.create(
            user=request.user,
            role=role or "",
            subject=form.cleaned_data["subject"],
            message=form.cleaned_data["message"],
            contact_email=form.cleaned_data["contact_email"],
        )
        messages.success(request, "Your message has been sent to the admin.")
        return redirect("users:overview")

    return render(request, "analytics_dashboard/contact_admin.html", {"form": form})


@login_required
@role_required("admin")
def support_messages(request):
    if request.method == "POST":
        message_item = get_object_or_404(SupportMessage, pk=request.POST.get("message_id"))
        new_status = request.POST.get("status")
        if new_status in {"new", "in_progress", "resolved"}:
            message_item.status = new_status
            message_item.seen_by_admin = True
            message_item.reviewed_at = timezone.now()
            message_item.save(update_fields=["status", "seen_by_admin", "reviewed_at"])
            messages.success(request, f"Updated message status for '{message_item.subject}'.")
        return redirect("analytics_dashboard:messages")

    inbox = SupportMessage.objects.select_related("user")
    return render(request, "analytics_dashboard/messages.html", {"messages_inbox": inbox})
