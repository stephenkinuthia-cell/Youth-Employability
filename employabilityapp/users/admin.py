from django.contrib import admin

from .models import EmployerProfile, GraduateProfile, UserProfile

admin.site.register(UserProfile)
admin.site.register(GraduateProfile)
admin.site.register(EmployerProfile)
