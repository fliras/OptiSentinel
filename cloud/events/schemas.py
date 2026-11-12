from enum import Enum

from pydantic import BaseModel, Field


class Severidade(str, Enum):
    baixa = "baixa"
    media = "media"
    alta = "alta"


class DescricaoEvento(BaseModel):

    resumo: str = Field(description="Uma frase descrevendo o que ocorre na imagem")
    descricaoDetalhada: str = Field(description="Descrição completa do evento")
    severidade: Severidade
    pessoasEnvolvidas: int = Field(ge=0, description="Quantidade de pessoas visíveis")
    riscosIdentificados: list[str] = Field(description="Lista de riscos detectados")
    confianca: float = Field(ge=0, le=1, description="Grau de confiança entre 0 e 1")