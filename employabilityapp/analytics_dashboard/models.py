from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone


class SkillDemandInsight(models.Model):
    skill_name = models.CharField(max_length=120, unique=True)
    demand_count = models.PositiveIntegerField(default=0)
    gap_count = models.PositiveIntegerField(default=0)
    source_label = models.CharField(max_length=100, default="platform")
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.skill_name


class AppVisit(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name="app_visits")
    role = models.CharField(max_length=20, blank=True)
    path = models.CharField(max_length=255)
    method = models.CharField(max_length=10)
    status_code = models.PositiveIntegerField(default=200)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.path} ({self.status_code})"


class SupportMessage(models.Model):
    STATUS_CHOICES = [
        ("new", "New"),
        ("in_progress", "In Progress"),
        ("resolved", "Resolved"),
    ]

    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name="support_messages")
    role = models.CharField(max_length=20, blank=True)
    subject = models.CharField(max_length=150)
    message = models.TextField()
    contact_email = models.EmailField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="new")
    seen_by_admin = models.BooleanField(default=False)
    reviewed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["status", "-created_at"]

    def __str__(self):
        return self.subject

    @property
    def user_status_text(self):
        if self.seen_by_admin and self.status == "resolved":
            return "Admin has seen this issue and marked it resolved."
        if self.seen_by_admin and self.status == "in_progress":
            return "Admin has seen this issue and is working on it."
        if self.seen_by_admin:
            return "Admin has seen this issue."
        return "Waiting for admin review."
