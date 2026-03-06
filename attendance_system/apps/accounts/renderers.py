from rest_framework.renderers import JSONRenderer


class CustomJSONRenderer(JSONRenderer):
    """
    Custom renderer wrapping all responses into:
    {"success": true/false, "data": ..., "message": "..."}
    """

    def render(self, data, accepted_media_type=None, renderer_context=None):
        response = renderer_context.get('response') if renderer_context else None

        # If data is already in our format, pass through
        if isinstance(data, dict) and 'success' in data:
            return super().render(data, accepted_media_type, renderer_context)

        if response and response.status_code >= 400:
            wrapped = {
                'success': False,
                'data': data,
                'message': self._extract_message(data) if data else 'An error occurred.',
            }
        else:
            wrapped = {
                'success': True,
                'data': data,
                'message': '',
            }
        return super().render(wrapped, accepted_media_type, renderer_context)

    @staticmethod
    def _extract_message(data):
        """Try to extract a meaningful error message from DRF error dicts."""
        if isinstance(data, dict):
            if 'detail' in data:
                return str(data['detail'])
            # Gather first error message from validation errors
            for key, value in data.items():
                if isinstance(value, list) and value:
                    return f"{key}: {value[0]}"
                elif isinstance(value, str):
                    return value
        if isinstance(data, list) and data:
            return str(data[0])
        return str(data)
