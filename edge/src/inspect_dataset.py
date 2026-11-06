"""
Etapa 1b - Conta anotacoes por classe de interesse, por split, lendo
diretamente os arquivos .txt de rotulo (formato YOLO), sem carregar imagens.

Uso:
    python src/inspect_dataset.py
"""

import sys
from collections import defaultdict
from pathlib import Path

from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import NEGATIVE_CLASS_IDS, ORIGINAL_CLASS_NAMES, RAW_DATASET_DIR, SPLITS, TARGET_CLASS_IDS

# Abaixo deste numero de anotacoes (em qualquer split), emitimos um alerta.
LIMIAR_ALERTA_BAIXA_CONTAGEM = 200


def contar_anotacoes_por_split(dataset_dir: Path) -> dict:
    """
    Retorna {split: {classe_id: contagem}} contando cada linha dos .txt de
    rotulo cujo primeiro numero esteja em TARGET_CLASS_IDS.
    """
    contagens = {split: defaultdict(int) for split in SPLITS}
    for split in SPLITS:
        labels_dir = dataset_dir / split / "labels"
        if not labels_dir.exists():
            print(f"[aviso] pasta de rotulos nao encontrada: {labels_dir}")
            continue
        for arquivo_txt in labels_dir.glob("*.txt"):
            with open(arquivo_txt, "r", encoding="utf-8") as f:
                for linha in f:
                    linha = linha.strip()
                    if not linha:
                        continue
                    classe_id = int(linha.split()[0])
                    if classe_id in TARGET_CLASS_IDS:
                        contagens[split][classe_id] += 1
    return contagens


def montar_tabela(contagens: dict) -> str:
    linhas = []
    for classe_id in TARGET_CLASS_IDS:
        nome = ORIGINAL_CLASS_NAMES[classe_id]
        linha = [f"{classe_id} ({nome})"]
        total = 0
        for split in SPLITS:
            n = contagens[split].get(classe_id, 0)
            linha.append(n)
            total += n
        linha.append(total)
        linhas.append(linha)
    cabecalho = ["Classe"] + SPLITS + ["Total"]
    return tabulate(linhas, headers=cabecalho, tablefmt="github")


def alertar_desbalanceamento(contagens: dict) -> None:
    print("\nAlertas de desbalanceamento:")
    algum_alerta = False
    for classe_id in TARGET_CLASS_IDS:
        nome = ORIGINAL_CLASS_NAMES[classe_id]
        total = sum(contagens[split].get(classe_id, 0) for split in SPLITS)
        if total < LIMIAR_ALERTA_BAIXA_CONTAGEM:
            algum_alerta = True
            tipo = " (classe NEGATIVA - ausencia de EPI)" if classe_id in NEGATIVE_CLASS_IDS else ""
            print(
                f"  [ALERTA] classe {classe_id} ({nome}){tipo}: apenas {total} anotacoes no total. "
                "Pode nao ser suficiente para o modelo aprender a detectar essa classe com recall aceitavel."
            )
        for split in SPLITS:
            n = contagens[split].get(classe_id, 0)
            if n == 0:
                algum_alerta = True
                print(f"  [ALERTA] classe {classe_id} ({nome}) tem ZERO anotacoes no split '{split}'.")
    if not algum_alerta:
        print("  Nenhum alerta critico encontrado (todas as classes tem contagem razoavel em todos os splits).")


def main() -> None:
    if not RAW_DATASET_DIR.exists():
        print(f"PARADO: {RAW_DATASET_DIR} nao existe. Rode primeiro: python src/download_dataset.py")
        sys.exit(1)

    contagens = contar_anotacoes_por_split(RAW_DATASET_DIR)
    print(f"Contagem de anotacoes por classe e split em: {RAW_DATASET_DIR}\n")
    print(montar_tabela(contagens))
    alertar_desbalanceamento(contagens)


if __name__ == "__main__":
    main()
