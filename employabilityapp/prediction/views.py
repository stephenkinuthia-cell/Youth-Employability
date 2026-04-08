from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.template.loader import render_to_string
from django.core.exceptions import PermissionDenied

from jobs.models import JobPosting
from recommendations.models import AIReport, CareerPathSuggestion, SkillRecommendation

from .forms import EmployabilityPredictionForm
from .models import EmployabilityAssessment
from .querysets import can_view_assessment, get_accessible_assessments
from .services import PredictionService


def home(request):
    latest_assessments = EmployabilityAssessment.objects.none()
    if request.user.is_authenticated:
        latest_assessments = get_accessible_assessments(request.user)[:5]
    context = {
        "assessment_count": EmployabilityAssessment.objects.count(),
        "job_count": JobPosting.objects.filter(is_active=True).count(),
        "latest_assessments": latest_assessments,
    }
    return render(request, "prediction/home.html", context)


@login_required
def dashboard(request):
    form = EmployabilityPredictionForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        service = PredictionService()
        result = service.predict_profile(form.cleaned_data)

        assessment = EmployabilityAssessment.objects.create(
            owner=request.user,
            graduate=getattr(request.user, "graduate_profile", None),
            full_name=form.cleaned_data["full_name"],
            email=form.cleaned_data["email"],
            degree_level=form.cleaned_data["degree_level"],
            field_of_study=form.cleaned_data["field_of_study"],
            region=form.cleaned_data["region"],
            graduation_year=form.cleaned_data["graduation_year"],
            experience_level=form.cleaned_data["experience_level"],
            current_skills=form.cleaned_data["skills"],
            average_starting_salary=form.cleaned_data["average_starting_salary"],
            skill_demand_score=form.cleaned_data["skill_demand_score"],
            employer_reputation_score=form.cleaned_data["employer_reputation_score"],
            remote_work_availability=form.cleaned_data["remote_work_availability"],
            employability_score=result.employability_score,
            risk_category=result.risk_category,
            model_status=result.model_status,
            strengths=result.strengths,
            weaknesses=result.weaknesses,
            recommendation_summary=" | ".join(result.report_next_steps),
        )

        for recommendation in result.recommendations:
            SkillRecommendation.objects.create(assessment=assessment, **recommendation)

        for career_path in result.career_paths:
            CareerPathSuggestion.objects.create(assessment=assessment, **career_path)

        AIReport.objects.create(
            assessment=assessment,
            strengths=result.strengths,
            weaknesses=result.weaknesses,
            next_steps=result.report_next_steps,
            report_markdown=result.report_markdown,
        )

        return redirect("prediction:assessment_detail", pk=assessment.pk)

    return render(request, "prediction/dashboard.html", {"form": form})


@login_required
def assessment_detail(request, pk):
    assessment = get_object_or_404(EmployabilityAssessment, pk=pk)
    if not can_view_assessment(request.user, assessment):
        raise PermissionDenied("You are not allowed to view this result.")

    matched_jobs = [
        job for job in JobPosting.objects.filter(is_active=True)
        if assessment.employability_score >= job.minimum_score
    ][:3]

    context = {
        "assessment": assessment,
        "recommendations": assessment.skill_recommendations.all(),
        "career_paths": assessment.career_paths.all(),
        "report": getattr(assessment, "ai_report", None),
        "matched_jobs": matched_jobs,
    }
    return render(request, "prediction/detail.html", context)


@login_required
def download_report(request, pk):
    assessment = get_object_or_404(EmployabilityAssessment, pk=pk)
    if not can_view_assessment(request.user, assessment):
        raise PermissionDenied("You are not allowed to download this report.")
    html = render_to_string(
        "prediction/report.html",
        {
            "assessment": assessment,
            "recommendations": assessment.skill_recommendations.all(),
            "career_paths": assessment.career_paths.all(),
            "report": getattr(assessment, "ai_report", None),
        },
        request=request,
    )
    response = HttpResponse(html, content_type="text/html")
    response["Content-Disposition"] = f'attachment; filename="ai-feedback-report-{assessment.pk}.html"'
    return response
