from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from events.serializers import EventRequestSerializer, EventResponseSerializer
from events.services.image_describer import (
    ImageDescriptionError,
    get_image_describer,
)

class DescribeEventView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        serializer = EventRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        image_file = serializer.validated_data["image"]
        image_bytes = image_file.read()
        metadata = serializer.to_metadata()

        try:
            describer = get_image_describer()
            evento = describer.describe(
                image_bytes,
                metadata,
                mime_type=image_file.content_type or "image/jpeg",
            )
        except ImageDescriptionError as exc:
            return Response(
                {"detail": str(exc)},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        payload = evento.model_dump(mode="json")

        return Response(payload, status=status.HTTP_200_OK)