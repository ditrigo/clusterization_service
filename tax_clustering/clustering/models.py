import uuid
from django.db import models

class Dataset(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    file = models.FileField(upload_to='datasets/')  # Файлы сохраняются в media/datasets/

    def __str__(self):
        return self.name

class ClusteringParameters(models.Model):
    job = models.OneToOneField('ClusteringJob', on_delete=models.CASCADE, related_name='parameters')
    feature_selection_method = models.CharField(max_length=100)
    dimensionality_reduction_method = models.CharField(max_length=100)
    clustering_algorithm = models.CharField(max_length=100)
    parameters = models.JSONField()

    def __str__(self):
        return f"Parameters for Job {self.job.id}"

class ClusteringJob(models.Model):
    STATUS_CHOICES = [
        ('Pending', 'Pending'),
        ('Processing', 'Processing'),
        ('Completed', 'Completed'),
        ('Failed', 'Failed'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='jobs')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='Pending')
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    result_file = models.FileField(upload_to='results/', null=True, blank=True)
    metrics = models.JSONField(null=True, blank=True)  # Поле для хранения метрик
    visualizations = models.JSONField(null=True, blank=True)  # Поле для хранения URL визуализаций

    def __str__(self):
        return f"Job {self.id} - {self.status}"
