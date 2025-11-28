# decorators.py
from django.shortcuts import redirect
from django.http import HttpResponseForbidden

def allowed_roles(roles=[]):
    def decorator(view_func):
        def wrapper(request, *args, **kwargs):
            if request.user.is_authenticated:
                if request.user.profile.role in roles:
                    return view_func(request, *args, **kwargs)
                else:
                    return HttpResponseForbidden("You are not allowed to access this page.")
            return redirect("login")
        return wrapper
    return decorator
