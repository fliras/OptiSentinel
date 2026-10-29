from enum import Enum
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

class Severidade(str, Enum):
    baixa = "baixa"
    media = "media"
    alta = "alta"

class DescricaoEvento(BaseModel):
    resumo: str = Field(description="Uma frase descrevendo o que ocorre na imagem")
    descricao_detalhada: str
    severidade: Severidade
    pessoas_envolvidas: int = Field(description="Quantidade de pessoas visíveis")
    riscos_identificados: list[str]
    confianca: float = Field(description="Entre 0 e 1")

client = genai.Client()  # lê GEMINI_API_KEY do ambiente

with open("frame.jpg", "rb") as f:
    imagem = f.read()

metadados = {
    "camera_id": "CAM-03",
    "setor": "Expedição",
    "timestamp": "2026-10-29T14:32:10",
    "deteccao": "pessoa_sem_capacete",
}

prompt = (
    "Descreva a situação ocorrida na imagem, considerando os metadados "
    f"da captura abaixo.\n\nMetadados: {metadados}"
)

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=[
        types.Part.from_bytes(data=imagem, mime_type="image/jpeg"),
        prompt,
    ],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=DescricaoEvento.model_json_schema(),
        temperature=0.2,
    ),
)

evento = DescricaoEvento.model_validate_json(response.text)
print(evento)