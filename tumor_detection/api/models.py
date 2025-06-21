from django.contrib.auth.models import AbstractUser
from django.db import models

# Create your models here.

class CustomUser(AbstractUser):
    email = models.EmailField(unique=True)
    age = models.PositiveIntegerField(null=True, blank=True)
    phone_number = models.CharField(max_length=20, blank=True)
    hospital_name = models.CharField(max_length=255, blank=True)
    hospital_department = models.CharField(max_length=255, blank=True)
    full_name = models.CharField(max_length=255, blank=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']  # username is still required by AbstractUser

    def __str__(self):
        return self.email
