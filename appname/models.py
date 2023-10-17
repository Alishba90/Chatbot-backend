from django.db import models

# Create your models here.
class Meta:
    permissions = [
        ("can_view_yourmodel", "Can view YourModel"),
        ("can_change_yourmodel", "Can change YourModel"),
        ("can_add_yourmodel", "Can add YourModel"),
        ("can_delete_yourmodel", "Can delete YourModel"),
    ]
class Chat(models.Model):

    query = models.CharField(max_length=255)
    response = models.CharField(max_length=255)