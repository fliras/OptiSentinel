from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.renderers import JSONRenderer, TemplateHTMLRenderer
from rest_framework.response import Response
from rest_framework.views import APIView
import time

from events.serializers import EventRequestSerializer
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

        return Response(evento.model_dump(mode="json"), status=status.HTTP_200_OK)
    
class DiagnoseEventView(APIView):
    """
    Executa o pipeline completo e devolve um relatório por etapa:
    HTTP → serializer → metadata → image → service_init → prompt
    → gemini → json_parse → schema_validation.

    Custo: uma chamada paga ao Gemini por POST.
    """

    parser_classes = [MultiPartParser, FormParser]
    renderer_classes = [JSONRenderer, TemplateHTMLRenderer]

    def get(self, request, *args, **kwargs):
        return Response(
            {"endpoint": request.path, "mode": "diagnose"},
            template_name="events/validate_form.html",
        )

    def post(self, request, *args, **kwargs):
        stages: list[dict] = []

        # 1. HTTP request
        received_fields = {
            k: str(v) for k, v in request.data.items()
            if not hasattr(v, "read")
        }
        received_files = {
            k: {
                "name": getattr(v, "name", None),
                "size": getattr(v, "size", None),
                "content_type": getattr(v, "content_type", None),
            }
            for k, v in request.data.items()
            if hasattr(v, "read")
        }
        stages.append({
            "name": "http_request",
            "ok": True,
            "data": {
                "method": request.method,
                "content_type": request.content_type,
                "received_fields": received_fields,
                "received_files": received_files,
            },
        })

        # 2. Serializer
        t0 = time.perf_counter()
        serializer = EventRequestSerializer(data=request.data)
        if not serializer.is_valid():
            stages.append({
                "name": "serializer",
                "ok": False,
                "duration_ms": (time.perf_counter() - t0) * 1000,
                "data": {"errors": serializer.errors},
            })
            return Response({"ok": False, "stages": stages},
                            status=status.HTTP_400_BAD_REQUEST)
        stages.append({
            "name": "serializer",
            "ok": True,
            "duration_ms": (time.perf_counter() - t0) * 1000,
            "data": {
                "validated_fields": {
                    k: str(v)
                    for k, v in serializer.validated_data.items()
                    if k != "image"
                },
            },
        })

        # 3. Metadata
        t0 = time.perf_counter()
        metadata = serializer.to_metadata()
        stages.append({
            "name": "metadata_transform",
            "ok": True,
            "duration_ms": (time.perf_counter() - t0) * 1000,
            "data": metadata,
        })

        # 4. Image read
        t0 = time.perf_counter()
        image_file = serializer.validated_data["image"]
        image_bytes = image_file.read()
        header = image_bytes[:8]
        stages.append({
            "name": "image_read",
            "ok": True,
            "duration_ms": (time.perf_counter() - t0) * 1000,
            "data": {
                "name": image_file.name,
                "size": image_file.size,
                "mime": image_file.content_type,
                "magic_header_hex": header.hex(),
                "looks_like_jpeg": header[:2] == b"\xff\xd8",
                "looks_like_png": header == b"\x89PNG\r\n\x1a\n",
            },
        })

        # 5. Service init
        t0 = time.perf_counter()
        try:
            describer = get_image_describer()
        except ImageDescriptionError as exc:
            stages.append({
                "name": "service_init",
                "ok": False,
                "duration_ms": (time.perf_counter() - t0) * 1000,
                "error": str(exc),
            })
            return Response({"ok": False, "stages": stages},
                            status=status.HTTP_502_BAD_GATEWAY)
        stages.append({
            "name": "service_init",
            "ok": True,
            "duration_ms": (time.perf_counter() - t0) * 1000,
            "data": {"model": describer.model},
        })

        # 6+. Etapas do service (prompt → gemini → json → schema)
        service_report = describer.describe_with_trace(
            image_bytes, metadata, image_file.content_type or "image/jpeg"
        )
        stages.extend(service_report["stages"])

        ok = service_report["ok"]
        return Response(
            {
                "ok": ok,
                "stages": stages,
                "result": service_report.get("result"),
            },
            status=status.HTTP_200_OK if ok else status.HTTP_502_BAD_GATEWAY,
        )


class ValidateEventInputView(APIView):
    """
    GET  → renderiza o formulário de teste no navegador.
    POST → valida o payload e devolve o JSON (sem chamar o Gemini).
    """

    parser_classes = [MultiPartParser, FormParser]
    renderer_classes = [JSONRenderer, TemplateHTMLRenderer]

    def get(self, request, *args, **kwargs):
        return Response(
            {"endpoint": request.path},
            template_name="events/validate_form.html",
        )

    def post(self, request, *args, **kwargs):
        serializer = EventRequestSerializer(data=request.data)

        received_fields = {
            k: str(v)
            for k, v in request.data.items()
            if not hasattr(v, "read")
        }
        received_files = {
            k: {
                "name": getattr(v, "name", None),
                "size": getattr(v, "size", None),
                "content_type": getattr(v, "content_type", None),
            }
            for k, v in request.data.items()
            if hasattr(v, "read")
        }

        if not serializer.is_valid():
            return Response(
                {
                    "ok": False,
                    "errors": serializer.errors,
                    "received": {
                        "fields": received_fields,
                        "files": received_files,
                    },
                    "hint": "Confira o formato esperado em EventRequestSerializer.",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        image_file = serializer.validated_data["image"]
        metadata = serializer.to_metadata()

        return Response(
            {
                "ok": True,
                "metadata": metadata,
                "image": {
                    "name": image_file.name,
                    "size": image_file.size,
                    "content_type": image_file.content_type,
                },
                "message": (
                    "Payload válido. Envie o mesmo conteúdo para "
                    "/api/v1/events/describe/ para chamar o Gemini."
                ),
            },
            status=status.HTTP_200_OK,
        )