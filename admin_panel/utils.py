# admin_panel/utils.py

from .models import Notification, Doctor, User
from django.utils import timezone

def create_notification(user, title, message):
    """Creates a notification for a user"""
    Notification.objects.create(
        user=user,
        title=title,
        message=message,
        timestamp=timezone.now()
    )

def create_admin_notification(title, message):
    """Creates a notification for all admins"""
    doctors = Doctor.objects.all()
    for doc in doctors:
        Notification.objects.create(
            user=doc.user,
            title=title,
            message=message,
            timestamp=timezone.now()
        )
#STOP CNTRL Z



