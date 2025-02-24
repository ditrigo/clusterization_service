# Generated by Django 4.2.17 on 2024-12-13 10:16

from django.db import migrations, models
import django.db.models.deletion
import uuid


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ClusteringJob',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('status', models.CharField(choices=[('Pending', 'Pending'), ('Processing', 'Processing'), ('Completed', 'Completed'), ('Failed', 'Failed')], default='Pending', max_length=20)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('completed_at', models.DateTimeField(blank=True, null=True)),
                ('result_file', models.FileField(blank=True, null=True, upload_to='results/')),
                ('metrics', models.JSONField(blank=True, null=True)),
                ('visualizations', models.JSONField(blank=True, null=True)),
                ('preprocessing_completed', models.BooleanField(default=False)),
                ('feature_selection_completed', models.BooleanField(default=False)),
                ('dimensionality_reduction_completed', models.BooleanField(default=False)),
                ('clustering_completed', models.BooleanField(default=False)),
                ('metrics_completed', models.BooleanField(default=False)),
                ('visualization_completed', models.BooleanField(default=False)),
                ('intermediate_data', models.FileField(blank=True, null=True, upload_to='intermediate/')),
                ('intermediate_files', models.JSONField(blank=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Dataset',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=255)),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
                ('file', models.FileField(upload_to='datasets/')),
            ],
        ),
        migrations.CreateModel(
            name='ClusteringParameters',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('feature_selection_method', models.CharField(max_length=100)),
                ('dimensionality_reduction_method', models.CharField(max_length=100)),
                ('clustering_algorithm', models.CharField(max_length=100)),
                ('parameters', models.JSONField()),
                ('job', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='parameters', to='clustering.clusteringjob')),
            ],
        ),
        migrations.AddField(
            model_name='clusteringjob',
            name='dataset',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='jobs', to='clustering.dataset'),
        ),
    ]
