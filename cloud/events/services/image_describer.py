import logging
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from django.conf import settings
from google import genai
from google.genai import types

from events.schemas import DescricaoEvento

logger = logging.getLogger(__name__)


class ImageDescriptionError(Exception):
    """Falha ao obter/validar a descrição da imagem."""

@dataclass
class StageResult:
    name: str
    ok: bool
    duration_ms: float = 0.0
    data: dict = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

class ImageDescriber:
    def __init__(self, client: genai.Client | None = None, model: str | None = None):
        api_key = getattr(settings, "GEMINI_API_KEY", None)
        if client is None and not api_key:
            raise ImageDescriptionError("GEMINI_API_KEY não configurada.")
        self._client = client or genai.Client(api_key=api_key)
        self._model = model or getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash")
        
    @property
    def model(self) -> str:
        return self._model

    def describe_with_trace(
        self,
        image_bytes: bytes,
        metadata: Mapping[str, Any],
        mime_type: str = "image/jpeg",
    ) -> dict:
        """
        Igual a describe(), mas devolve um relatório por etapa em vez de
        lançar. Falhas em uma etapa abortam as seguintes e ficam registradas
        no estágio onde ocorreram.
        """
        stages: list[StageResult] = []

        # 1. Montagem do prompt
        t0 = time.perf_counter()
        prompt = (
            "Descreva a situação ocorrida na imagem, considerando os metadados "
            f"da captura abaixo.\n\nMetadados: {dict(metadata)}"
        )
        stages.append(StageResult(
            name="prompt_assembly",
            ok=True,
            duration_ms=(time.perf_counter() - t0) * 1000,
            data={"prompt": prompt, "prompt_length": len(prompt)},
        ))

        # 2. Chamada ao Gemini
        t0 = time.perf_counter()
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
            stages.append(StageResult(
                name="gemini_call",
                ok=False,
                duration_ms=(time.perf_counter() - t0) * 1000,
                error=f"{type(exc).__name__}: {exc}",
            ))
            return {"ok": False, "stages": [s.to_dict() for s in stages]}

        raw_text = response.text or ""
        usage = getattr(response, "usage_metadata", None)
        usage_data = None
        if usage is not None:
            usage_data = {
                "prompt_tokens": getattr(usage, "prompt_token_count", None),
                "response_tokens": getattr(usage, "candidates_token_count", None),
                "total_tokens": getattr(usage, "total_token_count", None),
            }
        stages.append(StageResult(
            name="gemini_call",
            ok=True,
            duration_ms=(time.perf_counter() - t0) * 1000,
            data={
                "model": self._model,
                "raw_text_length": len(raw_text),
                "raw_text": raw_text if len(raw_text) <= 4000
                            else raw_text[:4000] + "...[truncated]",
                "usage": usage_data,
            },
        ))

        # 3. Parse JSON
        t0 = time.perf_counter()
        try:
            parsed = json.loads(raw_text)
        except Exception as exc:
            stages.append(StageResult(
                name="json_parse",
                ok=False,
                duration_ms=(time.perf_counter() - t0) * 1000,
                error=f"{type(exc).__name__}: {exc}",
                data={"raw_text_preview": raw_text[:1000]},
            ))
            return {"ok": False, "stages": [s.to_dict() for s in stages]}
        stages.append(StageResult(
            name="json_parse",
            ok=True,
            duration_ms=(time.perf_counter() - t0) * 1000,
            data={"keys": list(parsed.keys()) if isinstance(parsed, dict) else None},
        ))

        # 4. Validação contra o schema Pydantic
        t0 = time.perf_counter()
        try:
            evento = DescricaoEvento.model_validate(parsed)
        except Exception as exc:
            stages.append(StageResult(
                name="schema_validation",
                ok=False,
                duration_ms=(time.perf_counter() - t0) * 1000,
                error=f"{type(exc).__name__}: {exc}",
                data={"parsed": parsed},
            ))
            return {"ok": False, "stages": [s.to_dict() for s in stages]}
        stages.append(StageResult(
            name="schema_validation",
            ok=True,
            duration_ms=(time.perf_counter() - t0) * 1000,
            data=evento.model_dump(mode="json"),
        ))

        return {
            "ok": True,
            "stages": [s.to_dict() for s in stages],
            "result": evento.model_dump(mode="json"),
        }

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