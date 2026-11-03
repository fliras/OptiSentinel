from enum import Enum

from pydantic import BaseModel, Field


class Severidade(str, Enum):
    baixa = "baixa"
    media = "media"
    alta = "alta"


class DescricaoEvento(BaseModel):
    resumo: str = Field(description="Uma frase descrevendo o que ocorre na imagem")
    descricao_detalhada: str
    severidade: Severidade
    pessoas_envolvidas: int = Field(
        ge=0,
        description="Quantidade de pessoas visíveis",
    )
    riscos_identificados: list[str]
    confianca: float = Field(
        ge=0,
        le=1,
        description="Grau de confiança entre 0 e 1",
    )
