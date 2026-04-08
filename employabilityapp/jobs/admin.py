from django.contrib import admin

from .models import JobApplication, JobPosting

admin.site.register(JobPosting)
admin.site.register(JobApplication)
