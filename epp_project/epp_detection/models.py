from django.db import models

# Create your models here.
# ppe_detection/models.py
from django.db import models

class Incident(models.Model):
    datetime = models.DateTimeField(auto_now_add=True)
    camera = models.CharField(max_length=100)
    incident_type = models.CharField(max_length=200)  # Ej: "Sin casco", "Sin chaleco"

    def __str__(self):
        return f"{self.datetime} - {self.incident_type}"