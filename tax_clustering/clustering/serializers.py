from rest_framework import serializers
from .models import Dataset, ClusteringJob, ClusteringParameters
import pandas as pd

class DatasetSerializer(serializers.ModelSerializer):
    file_url = serializers.SerializerMethodField()
    columns = serializers.SerializerMethodField()
    preview = serializers.SerializerMethodField()

    class Meta:
        model = Dataset
        fields = ['id', 'name', 'uploaded_at', 'file', 'file_url', 'columns', 'preview']

    def get_file_url(self, obj):
        request = self.context.get('request')
        if obj.file and hasattr(obj.file, 'url'):
            return request.build_absolute_uri(obj.file.url)
        return None

    def get_columns(self, obj):
        try:
            # Считываем только первую строку для получения заголовков
            df = pd.read_csv(obj.file.path, nrows=0)
            return list(df.columns)
        except Exception as e:
            return []

    def get_preview(self, obj):
        try:
            # Считываем первые 5 строк
            df = pd.read_csv(obj.file.path, nrows=5)
            # Преобразуем DataFrame в список словарей
            return df.to_dict(orient='records')
        except Exception as e:
            return []

class ClusteringParametersSerializer(serializers.ModelSerializer):
    feature_selection = serializers.DictField(write_only=True, required=False)
    dimensionality_reduction = serializers.DictField(write_only=True, required=False)
    clustering = serializers.DictField(write_only=True, required=False)

    class Meta:
        model = ClusteringParameters
        fields = [
            'feature_selection_method',
            'dimensionality_reduction_method',
            'clustering_algorithm',
            'feature_selection',
            'dimensionality_reduction',
            'clustering'
        ]
        read_only_fields = []

    def create(self, validated_data):
        feature_selection = validated_data.pop('feature_selection', {})
        dimensionality_reduction = validated_data.pop('dimensionality_reduction', {})
        clustering = validated_data.pop('clustering', {})
        parameters = {
            'feature_selection': feature_selection,
            'dimensionality_reduction': dimensionality_reduction,
            'clustering': clustering
        }
        validated_data['parameters'] = parameters
        return super().create(validated_data)

class ClusteringJobSerializer(serializers.ModelSerializer):
    # Добавляем новое поле для названия датасета
    dataset_name = serializers.CharField(source="dataset.name", read_only=True)
    parameters = ClusteringParametersSerializer()
    result_file_url = serializers.SerializerMethodField()
    metrics = serializers.JSONField(read_only=True)
    visualizations = serializers.JSONField(read_only=True)

    # Поля для отслеживания стадий обработки
    preprocessing_completed = serializers.BooleanField(read_only=True)
    feature_selection_completed = serializers.BooleanField(read_only=True)
    dimensionality_reduction_completed = serializers.BooleanField(read_only=True)
    clustering_completed = serializers.BooleanField(read_only=True)
    metrics_completed = serializers.BooleanField(read_only=True)
    visualization_completed = serializers.BooleanField(read_only=True)

    class Meta:
        model = ClusteringJob
        fields = [
            'id',
            'dataset',         # По-прежнему возвращается id датасета
            'dataset_name',    # А теперь добавляем и его название
            'status',
            'created_at',
            'completed_at',
            'result_file',
            'result_file_url',
            'parameters',
            'metrics',
            'visualizations',
            'preprocessing_completed',
            'feature_selection_completed',
            'dimensionality_reduction_completed',
            'clustering_completed',
            'metrics_completed',
            'visualization_completed',
        ]
        read_only_fields = [
            'status', 'created_at', 'completed_at', 'result_file',
            'metrics', 'visualizations',
            'preprocessing_completed',
            'feature_selection_completed',
            'dimensionality_reduction_completed',
            'clustering_completed',
            'metrics_completed',
            'visualization_completed',
        ]

    def get_result_file_url(self, obj):
        request = self.context.get('request')
        if obj.result_file and hasattr(obj.result_file, 'url'):
            return request.build_absolute_uri(obj.result_file.url)
        return None

    def create(self, validated_data):
        parameters_data = validated_data.pop('parameters')
        feature_selection = parameters_data.pop('feature_selection', {})
        dimensionality_reduction = parameters_data.pop('dimensionality_reduction', {})
        clustering = parameters_data.pop('clustering', {})
        job = ClusteringJob.objects.create(**validated_data)
        ClusteringParameters.objects.create(
            job=job,
            feature_selection_method=parameters_data.get('feature_selection_method'),
            dimensionality_reduction_method=parameters_data.get('dimensionality_reduction_method'),
            clustering_algorithm=parameters_data.get('clustering_algorithm'),
            parameters={
                'feature_selection': feature_selection,
                'dimensionality_reduction': dimensionality_reduction,
                'clustering': clustering
            }
        )
        return job