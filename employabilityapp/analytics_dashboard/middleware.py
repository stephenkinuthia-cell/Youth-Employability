from .models import AppVisit


class AppUsageTrackingMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)

        if request.path.startswith("/static/") or request.path.startswith("/favicon"):
            return response

        user = request.user if getattr(request, "user", None) and request.user.is_authenticated else None
        role = ""
        if user and hasattr(user, "platform_profile"):
            role = user.platform_profile.role

        AppVisit.objects.create(
            user=user,
            role=role,
            path=request.path[:255],
            method=request.method,
            status_code=response.status_code,
        )
        return response
