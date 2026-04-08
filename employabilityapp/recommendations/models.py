from django.db import models

from prediction.models import EmployabilityAssessment


class SkillRecommendation(models.Model):
    assessment = models.ForeignKey(
        EmployabilityAssessment,
        on_delete=models.CASCADE,
        related_name="skill_recommendations",
    )
    title = models.CharField(max_length=150)
    category = models.CharField(max_length=50)
    rationale = models.TextField()
    priority = models.CharField(max_length=20, default="Medium")
    improvement_potential = models.FloatField(default=0)
    course_name = models.CharField(max_length=150, blank=True)
    course_url = models.URLField(blank=True)

    def __str__(self):
        return self.title


class CareerPathSuggestion(models.Model):
    assessment = models.ForeignKey(
        EmployabilityAssessment,
        on_delete=models.CASCADE,
        related_name="career_paths",
    )
    title = models.CharField(max_length=120)
    match_score = models.FloatField(default=0)
    summary = models.TextField()

    def __str__(self):
        return self.title


class AIReport(models.Model):
    assessment = models.OneToOneField(
        EmployabilityAssessment,
        on_delete=models.CASCADE,
        related_name="ai_report",
    )
    strengths = models.JSONField(default=list, blank=True)
    weaknesses = models.JSONField(default=list, blank=True)
    next_steps = models.JSONField(default=list, blank=True)
    report_markdown = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"AI Report for assessment {self.assessment_id}"
