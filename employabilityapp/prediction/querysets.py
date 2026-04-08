from prediction.models import EmployabilityAssessment
from users.permissions import get_user_role


def get_accessible_assessments(user):
    role = get_user_role(user)
    if role in {"admin", "employer"}:
        return EmployabilityAssessment.objects.select_related("owner", "graduate", "graduate__user")
    return EmployabilityAssessment.objects.filter(owner=user).select_related("owner", "graduate", "graduate__user")


def can_view_assessment(user, assessment):
    role = get_user_role(user)
    if role in {"admin", "employer"}:
        return True
    return assessment.owner_id == user.id
