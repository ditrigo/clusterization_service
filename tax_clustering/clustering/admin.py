from django.contrib import admin
from .models import Dataset, ClusteringJob, ClusteringParameters

admin.site.register(Dataset)
admin.site.register(ClusteringJob)
admin.site.register(ClusteringParameters)
