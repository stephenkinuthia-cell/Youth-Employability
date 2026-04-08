from django.contrib.auth.models import User
from django.db import models

from prediction.models import EmployabilityAssessment


class JobPosting(models.Model):
    employer = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name="job_postings")
    title = models.CharField(max_length=120)
    industry = models.CharField(max_length=80)
    location = models.CharField(max_length=120)
    description = models.TextField()
    required_skills = models.JSONField(default=list, blank=True)
    minimum_score = models.FloatField(default=55)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.title


class JobApplication(models.Model):
    STATUS_CHOICES = [
        ("draft", "Draft"),
        ("submitted", "Submitted"),
        ("shortlisted", "Shortlisted"),
        ("rejected", "Rejected"),
    ]

    job = models.ForeignKey(JobPosting, on_delete=models.CASCADE, related_name="applications")
    assessment = models.ForeignKey(
        EmployabilityAssessment,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="job_applications",
    )
    full_name = models.CharField(max_length=120)
    email = models.EmailField()
    matched_score = models.FloatField(default=0)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="draft")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.full_name} -> {self.job.title}"
