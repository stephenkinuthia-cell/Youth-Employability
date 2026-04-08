from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from prediction.querysets import get_accessible_assessments


@login_required
def recommendation_list(request):
    assessments = get_accessible_assessments(request.user).prefetch_related("skill_recommendations", "career_paths")[:10]
    return render(request, "recommendations/list.html", {"assessments": assessments})
