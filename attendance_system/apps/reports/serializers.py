from rest_framework import serializers


class MonthlyReportQuerySerializer(serializers.Serializer):
    month = serializers.IntegerField(min_value=1, max_value=12)
    year = serializers.IntegerField(min_value=2020)
    department_id = serializers.IntegerField(required=False)


class ExportQuerySerializer(serializers.Serializer):
    month = serializers.IntegerField(min_value=1, max_value=12)
    year = serializers.IntegerField(min_value=2020)
    department_id = serializers.IntegerField(required=False)
