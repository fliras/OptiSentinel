import logging
from typing import Any, Mapping

from django.conf import settings
from google import genai
from google.genai import types

from events.schemas import DescricaoEvento

logger = logging.getLogger(__name__)


class ImageDescriptionError(Exception):
    """Falha ao obter/validar a descrição da imagem."""


class ImageDescriber:
    def __init__(self, client: genai.Client | None = None, model: str | None = None):
        api_key = getattr(settings, "GEMINI_API_KEY", None)
        if client is None and not api_key:
            raise ImageDescriptionError("GEMINI_API_KEY não configurada.")
        self._client = client or genai.Client(api_key=api_key)
        self._model = model or getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash")

    def describe(
        self,
        image_bytes: bytes,
        metadata: Mapping[str, Any],
        mime_type: str = "image/jpeg",
    ) -> DescricaoEvento:
        prompt = (
            "Descreva a situação ocorrida na imagem, considerando os metadados "
            f"da captura abaixo.\n\nMetadados: {dict(metadata)}"
        )

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=DescricaoEvento.model_json_schema(),
                    temperature=0.2,
                ),
            )
        except Exception as exc:
            logger.exception("Falha na chamada ao Gemini")
            raise ImageDescriptionError(f"Erro ao consultar o Gemini: {exc}") from exc

        if not response.text:
            raise ImageDescriptionError("Resposta vazia do modelo.")

        try:
            return DescricaoEvento.model_validate_json(response.text)
        except Exception as exc:
            logger.exception("JSON retornado pelo Gemini não adere ao schema")
            raise ImageDescriptionError(f"Resposta inválida do modelo: {exc}") from exc


_describer_singleton: ImageDescriber | None = None


def get_image_describer() -> ImageDescriber:
    """Instancia única (o Client do GenAI é caro para construir a cada request)."""
    global _describer_singleton
    if _describer_singleton is None:
        _describer_singleton = ImageDescriber()
    return _describer_singleton