from django.contrib.auth.models import User
from django.db import models

from employabilityapp.platform_choices import DEGREE_LEVEL_CHOICES, EXPERIENCE_LEVEL_CHOICES, FIELD_OF_STUDY_CHOICES, REGION_CHOICES
from users.models import GraduateProfile


class EmployabilityAssessment(models.Model):
    RISK_CHOICES = [
        ("Low", "Low"),
        ("Medium", "Medium"),
        ("High", "High"),
    ]

    owner = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="employability_assessments",
        null=True,
        blank=True,
    )
    graduate = models.ForeignKey(
        GraduateProfile,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="assessments",
    )
    full_name = models.CharField(max_length=120)
    email = models.EmailField()
    degree_level = models.CharField(max_length=30, choices=DEGREE_LEVEL_CHOICES)
    field_of_study = models.CharField(max_length=80, choices=FIELD_OF_STUDY_CHOICES)
    region = models.CharField(max_length=50, choices=REGION_CHOICES)
    graduation_year = models.PositiveIntegerField()
    experience_level = models.CharField(max_length=20, choices=EXPERIENCE_LEVEL_CHOICES)
    current_skills = models.JSONField(default=list, blank=True)
    average_starting_salary = models.PositiveIntegerField(default=0)
    skill_demand_score = models.PositiveIntegerField(default=50)
    employer_reputation_score = models.PositiveIntegerField(default=50)
    remote_work_availability = models.PositiveIntegerField(default=50)
    employability_score = models.FloatField()
    risk_category = models.CharField(max_length=20, choices=RISK_CHOICES)
    model_status = models.CharField(max_length=120, default="demo_fallback")
    strengths = models.JSONField(default=list, blank=True)
    weaknesses = models.JSONField(default=list, blank=True)
    recommendation_summary = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.full_name} - {self.employability_score:.1f}%"
