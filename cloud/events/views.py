
import json
import logging
import time
from typing import Any

from django.conf import settings
from google import genai
from google.genai import types
from rest_framework import serializers, status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.renderers import JSONRenderer, TemplateHTMLRenderer
from rest_framework.response import Response
from rest_framework.views import APIView

from events.schemas import DescricaoEvento

logger = logging.getLogger(__name__)

MIMES_ACEITOS = {"image/jpeg", "image/png"}
TAMANHO_MAXIMO_IMAGEM = 10 * 1024 * 1024
MODELO_PADRAO = "gemini-2.5-flash-lite"


class ErroDescricao(Exception):
    """Qualquer falha recuperável no pipeline (chave, Gemini, schema)."""



class RequisicaoEvento(serializers.Serializer):
    imagem = serializers.ImageField()
    idCamera = serializers.CharField(max_length=64)
    setor = serializers.CharField(max_length=128)
    momento = serializers.DateTimeField()
    deteccao = serializers.CharField(max_length=128)

    def validate_imagem(self, valor):
        if valor.size > TAMANHO_MAXIMO_IMAGEM:
            limite = TAMANHO_MAXIMO_IMAGEM // (1024 * 1024)
            raise serializers.ValidationError(f"Imagem excede {limite} MB.")
        if valor.content_type not in MIMES_ACEITOS:
            raise serializers.ValidationError(
                f"Formato não suportado: {valor.content_type}. Use JPEG ou PNG."
            )
        return valor

    def paraMetadados(self) -> dict:
        d = self.validated_data
        return {
            "idCamera": d["idCamera"],
            "setor": d["setor"],
            "momento": d["momento"].isoformat(),
            "deteccao": d["deteccao"],
        }


_clienteGemini: genai.Client | None = None


def obterClienteGemini() -> genai.Client:
    global _clienteGemini
    if _clienteGemini is None:
        chave = getattr(settings, "GEMINI_API_KEY", None)
        if not chave:
            raise ErroDescricao("GEMINI_API_KEY não configurada no ambiente.")
        logger.info("Inicializando cliente Gemini")
        _clienteGemini = genai.Client(api_key=chave)
    return _clienteGemini


def obterModelo() -> str:
    return getattr(settings, "GEMINI_MODEL", MODELO_PADRAO)



def montarPrompt(metadados: dict) -> str:
    return (
        "Descreva a situação ocorrida na imagem, considerando os metadados "
        f"da captura abaixo.\n\nMetadados: {metadados}"
    )


def chamarGemini(imagemBytes: bytes, mime: str, prompt: str) -> tuple[str, dict | None]:
    """Executa a chamada ao Gemini e devolve (textoBruto, dadosDeUso)."""
    cliente = obterClienteGemini()
    modelo = obterModelo()

    logger.info("Chamando Gemini: modelo=%s mime=%s bytes=%d", modelo, mime, len(imagemBytes))
    t0 = time.perf_counter()

    try:
        resposta = cliente.models.generate_content(
            model=modelo,
            contents=[
                types.Part.from_bytes(data=imagemBytes, mime_type=mime),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=DescricaoEvento.model_json_schema(),
                temperature=0.2,
            ),
        )
    except Exception as exc:
        logger.exception("Chamada ao Gemini falhou")
        raise ErroDescricao(f"Erro ao consultar o Gemini: {exc}") from exc

    duracao = (time.perf_counter() - t0) * 1000
    texto = resposta.text or ""
    uso = getattr(resposta, "usage_metadata", None)
    dadosUso = None
    if uso is not None:
        dadosUso = {
            "tokensPrompt": getattr(uso, "prompt_token_count", None),
            "tokensResposta": getattr(uso, "candidates_token_count", None),
            "tokensTotal": getattr(uso, "total_token_count", None),
        }

    logger.info(
        "Gemini respondeu em %.0f ms (%d chars, %s tokens)",
        duracao,
        len(texto),
        dadosUso["tokensTotal"] if dadosUso else "?",
    )
    return texto, dadosUso


def descreverImagem(imagemBytes: bytes, mime: str, metadados: dict) -> DescricaoEvento:
   
    prompt = montarPrompt(metadados)
    texto, _ = chamarGemini(imagemBytes, mime, prompt)

    if not texto:
        raise ErroDescricao("O modelo devolveu resposta vazia.")

    try:
        dados = json.loads(texto)
    except json.JSONDecodeError as exc:
        logger.error("JSON inválido do Gemini: %s | prévia=%r", exc, texto[:300])
        raise ErroDescricao(f"Resposta do modelo não é JSON válido: {exc}") from exc

    try:
        evento = DescricaoEvento.model_validate(dados)
    except Exception as exc:
        logger.error("Schema inválido: %s | dados=%r", exc, dados)
        raise ErroDescricao(f"Resposta não adere ao schema: {exc}") from exc

    logger.info(
        "Descrição pronta: severidade=%s confiança=%.2f pessoas=%d riscos=%d",
        evento.severidade.value,
        evento.confianca,
        evento.pessoasEnvolvidas,
        len(evento.riscosIdentificados),
    )
    return evento



def dadosRecebidos(request) -> dict:
    
    campos = {k: str(v) for k, v in request.data.items() if not hasattr(v, "read")}
    arquivos = {
        k: {
            "nome": getattr(v, "name", None),
            "tamanho": getattr(v, "size", None),
            "mime": getattr(v, "content_type", None),
        }
        for k, v in request.data.items()
        if hasattr(v, "read")
    }
    return {"campos": campos, "arquivos": arquivos}


def medirEtapa(etapas: list, nome: str, fn, dados: dict | None = None):
    
    t0 = time.perf_counter()
    try:
        resultado = fn()
    except Exception as exc:
        etapas.append({
            "nome": nome,
            "ok": False,
            "duracaoMs": round((time.perf_counter() - t0) * 1000, 2),
            "erro": f"{type(exc).__name__}: {exc}",
        })
        raise
    etapas.append({
        "nome": nome,
        "ok": True,
        "duracaoMs": round((time.perf_counter() - t0) * 1000, 2),
        "dados": dados or {},
    })
    return resultado


class DescreverEventoView(APIView):
    parser_classes = [MultiPartParser, FormParser]
    renderer_classes = [JSONRenderer, TemplateHTMLRenderer]

    def get(self, request):
        return Response({"endpoint": request.path}, template_name="events/formulario.html")

    def post(self, request):
        logger.info("POST %s — payload recebido", request.path)

        serializer = RequisicaoEvento(data=request.data)
        if not serializer.is_valid():
            logger.warning("Payload inválido: %s", serializer.errors)
            return Response(
                {
                    "erro": "Payload inválido",
                    "detalhes": serializer.errors,
                    "recebido": dadosRecebidos(request),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        metadados = serializer.paraMetadados()
        logger.info("Payload válido: %s", metadados)

        imagem = serializer.validated_data["imagem"]
        try:
            evento = descreverImagem(
                imagem.read(),
                imagem.content_type or "image/jpeg",
                metadados,
            )
        except ErroDescricao as exc:
            logger.error("Falha ao descrever: %s", exc)
            return Response({"erro": str(exc)}, status=status.HTTP_502_BAD_GATEWAY)

        return Response(evento.model_dump(mode="json"), status=status.HTTP_200_OK)


# ---------------------------------------------------------------------------
# View de diagnóstico: POST /api/v1/events/diagnosticar/
# ---------------------------------------------------------------------------
class DiagnosticarEventoView(APIView):
    """Executa o pipeline inteiro e devolve um relatório por etapa.

    Consome cota do Gemini (uma chamada por POST). Pensada para
    desenvolvimento e para a banca — não é o endpoint de produção.
    """

    parser_classes = [MultiPartParser, FormParser]
    renderer_classes = [JSONRenderer, TemplateHTMLRenderer]

    def get(self, request):
        return Response({"endpoint": request.path}, template_name="events/formulario.html")

    def post(self, request):
        etapas: list[dict] = []
        logger.info("POST %s — diagnóstico iniciado", request.path)

        # 1. Payload cru
        etapas.append({
            "nome": "payload_recebido",
            "ok": True,
            "duracaoMs": 0.0,
            "dados": dadosRecebidos(request),
        })

        # 2. Validação
        serializer = RequisicaoEvento(data=request.data)
        try:
            medirEtapa(etapas, "validacao", lambda: serializer.is_valid(raise_exception=True))
        except Exception as exc:
            logger.warning("Diagnóstico: validação falhou — %s", exc)
            return Response(
                {"ok": False, "etapas": etapas, "erros": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST,
            )

      # 3. Metadados
        metadados = medirEtapa(etapas, "metadados", serializer.paraMetadados)
        etapas[-1]["dados"] = metadados

        # 4. Imagem
        imagem = serializer.validated_data["imagem"]
        imagemBytes = medirEtapa(etapas, "imagem_lida", imagem.read)
        cabecalho = imagemBytes[:8]
        etapas[-1]["dados"] = {
            "nome": imagem.name,
            "tamanho": imagem.size,
            "mime": imagem.content_type,
            "cabecalhoHex": cabecalho.hex(),
            "eJpeg": cabecalho[:2] == b"\xff\xd8",
            "ePng": cabecalho == b"\x89PNG\r\n\x1a\n",
        }

        # 5. Cliente Gemini
        try:
            medirEtapa(etapas, "cliente_gemini", obterClienteGemini)
            etapas[-1]["dados"] = {"modelo": obterModelo()}
        except ErroDescricao as exc:
            logger.error("Diagnóstico: cliente Gemini falhou — %s", exc)
            return Response({"ok": False, "etapas": etapas}, status=status.HTTP_502_BAD_GATEWAY)

        # 6. Prompt
        prompt = medirEtapa(etapas, "prompt", lambda: montarPrompt(metadados))
        etapas[-1]["dados"] = {"texto": prompt, "tamanho": len(prompt)}

        # 7. Chamada ao Gemini
        try:
            texto, uso = medirEtapa(
                etapas,
                "gemini",
                lambda: chamarGemini(imagemBytes, imagem.content_type or "image/jpeg", prompt),
            )
        except ErroDescricao as exc:
            return Response({"ok": False, "etapas": etapas}, status=status.HTTP_502_BAD_GATEWAY)
        etapas[-1]["dados"] = {
            "modelo": obterModelo(),
            "tamanhoTexto": len(texto),
            "texto": texto if len(texto) <= 4000 else texto[:4000] + "...[truncado]",
            "uso": uso,
        }

        # 8. Parse JSON
        try:
            dados = medirEtapa(etapas, "json", lambda: json.loads(texto))
        except Exception as exc:
            etapas[-1]["dados"] = {"previa": texto[:1000]}
            return Response({"ok": False, "etapas": etapas}, status=status.HTTP_502_BAD_GATEWAY)
        etapas[-1]["dados"] = {"chaves": list(dados.keys()) if isinstance(dados, dict) else None}

        # 9. Schema
        try:
            evento = medirEtapa(etapas, "schema", lambda: DescricaoEvento.model_validate(dados))
        except Exception as exc:
            etapas[-1]["dados"] = {"dados": dados}
            return Response({"ok": False, "etapas": etapas}, status=status.HTTP_502_BAD_GATEWAY)
        etapas[-1]["dados"] = evento.model_dump(mode="json")

        logger.info("Diagnóstico concluído com sucesso (%d etapas)", len(etapas))
        return Response({
            "ok": True,
            "etapas": etapas,
            "resultado": evento.model_dump(mode="json"),
        })