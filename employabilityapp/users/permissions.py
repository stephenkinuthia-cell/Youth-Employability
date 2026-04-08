from functools import wraps

from django.contrib import messages
from django.shortcuts import redirect


def get_user_role(user):
    if not user.is_authenticated:
        return None
    profile = getattr(user, "platform_profile", None)
    return getattr(profile, "role", None)


def role_required(*allowed_roles):
    def decorator(view_func):
        @wraps(view_func)
        def _wrapped_view(request, *args, **kwargs):
            if not request.user.is_authenticated:
                return redirect("users:login")

            role = get_user_role(request.user)
            if role not in allowed_roles:
                messages.error(request, "You do not have permission to access that page.")
                return redirect("prediction:dashboard")

            return view_func(request, *args, **kwargs)

        return _wrapped_view

    return decorator
