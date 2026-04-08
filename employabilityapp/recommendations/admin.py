from django.contrib import admin

from .models import AIReport, CareerPathSuggestion, SkillRecommendation

admin.site.register(SkillRecommendation)
admin.site.register(CareerPathSuggestion)
admin.site.register(AIReport)
