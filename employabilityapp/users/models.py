from django.contrib.auth.models import User
from django.db import models

from employabilityapp.platform_choices import DEGREE_LEVEL_CHOICES, EXPERIENCE_LEVEL_CHOICES, FIELD_OF_STUDY_CHOICES


class UserProfile(models.Model):
    ROLE_CHOICES = [
        ("student", "Student / Graduate"),
        ("employer", "Employer"),
        ("admin", "Administrator"),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="platform_profile")
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default="student")
    phone_number = models.CharField(max_length=30, blank=True)
    location = models.CharField(max_length=120, blank=True)
    bio = models.TextField(blank=True)
    admin_code_used = models.CharField(max_length=40, blank=True)

    def __str__(self):
        return f"{self.user.username} ({self.role})"


class GraduateProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="graduate_profile")
    degree_level = models.CharField(max_length=30, choices=DEGREE_LEVEL_CHOICES)
    field_of_study = models.CharField(max_length=80, choices=FIELD_OF_STUDY_CHOICES)
    graduation_year = models.PositiveIntegerField()
    experience_level = models.CharField(max_length=20, choices=EXPERIENCE_LEVEL_CHOICES, default="entry")
    skills_text = models.TextField(blank=True, help_text="Comma-separated skills")
    career_goal = models.CharField(max_length=150, blank=True)

    def __str__(self):
        return f"{self.user.get_full_name() or self.user.username} - {self.field_of_study}"


class EmployerProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="employer_profile")
    company_name = models.CharField(max_length=150)
    industry = models.CharField(max_length=120)
    website = models.URLField(blank=True)
    company_size = models.CharField(max_length=50, blank=True)
    contact_person = models.CharField(max_length=120, blank=True)

    def __str__(self):
        return self.company_name
