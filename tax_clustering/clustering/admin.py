from django.contrib import admin

from .models import ClusteringJob, ClusteringParameters, Dataset

admin.site.register(Dataset)
admin.site.register(ClusteringJob)
admin.site.register(ClusteringParameters)
