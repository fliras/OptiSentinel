"""
Etapa 1a - Download do dataset PPE (Kaggle) e geracao de um data.yaml local
com caminhos absolutos.

Uso:
    python src/download_dataset.py

Requer credenciais do Kaggle configuradas ANTES de rodar (o script nao
grava nem pede credenciais em texto - ele so verifica se ja existem):
  Opcao 1 (recomendada): baixe kaggle.json em https://www.kaggle.com/settings
      -> "Create New Token", e salve em ~/.kaggle/kaggle.json (chmod 600).
  Opcao 2: exporte variaveis de ambiente antes de rodar o script:
      export KAGGLE_USERNAME="seu_usuario"
      export KAGGLE_KEY="sua_chave"
Nunca commitar essas credenciais. O .gitignore do projeto ja exclui
~/.kaggle e kaggle.json, mas o arquivo em si deve ficar fora do repositorio.
"""

import os
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import KAGGLE_DATASET_ID, ORIGINAL_CLASS_NAMES, RAW_DATASET_DIR, SPLITS


def credenciais_kaggle_presentes() -> bool:
    """Verifica se ha credenciais do Kaggle configuradas, sem exigir nenhuma delas."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    kaggle_json = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    return kaggle_json.exists()


def encontrar_pasta_dataset(base_path: Path) -> Path:
    """
    kagglehub as vezes baixa o conteudo dentro de uma subpasta aninhada.
    Procura o primeiro diretorio que contenha train/valid/test.
    """
    candidatos = [base_path] + [p for p in base_path.rglob("*") if p.is_dir()]
    for candidato in candidatos:
        if all((candidato / split).exists() for split in SPLITS):
            return candidato
    raise FileNotFoundError(
        f"Nao encontrei pastas train/valid/test dentro de {base_path}. "
        "Verifique a estrutura baixada manualmente."
    )


def gerar_data_yaml_local(dataset_dir: Path, destino: Path) -> Path:
    """Gera um data.yaml com caminhos ABSOLUTOS, preservando nc=14 e os nomes originais."""
    data = {
        "path": str(dataset_dir),
        "train": str((dataset_dir / "train" / "images").resolve()),
        "val": str((dataset_dir / "valid" / "images").resolve()),
        "test": str((dataset_dir / "test" / "images").resolve()),
        "nc": len(ORIGINAL_CLASS_NAMES),
        "names": [ORIGINAL_CLASS_NAMES[i] for i in range(len(ORIGINAL_CLASS_NAMES))],
    }
    destino.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return destino


def main() -> None:
    if not credenciais_kaggle_presentes():
        print(
            "PARADO: nao encontrei credenciais do Kaggle.\n\n"
            "Configure uma das opcoes abaixo e rode este script novamente:\n\n"
            "  1) Baixe kaggle.json em https://www.kaggle.com/settings "
            "(botao 'Create New Token') e salve em ~/.kaggle/kaggle.json\n"
            "     depois rode: chmod 600 ~/.kaggle/kaggle.json\n\n"
            "  2) Ou exporte variaveis de ambiente antes de rodar:\n"
            "     export KAGGLE_USERNAME=\"seu_usuario\"\n"
            "     export KAGGLE_KEY=\"sua_chave\"\n"
        )
        sys.exit(1)

    import kagglehub  # importado aqui para nao falhar antes da checagem de credenciais

    print(f"Baixando dataset '{KAGGLE_DATASET_ID}' via kagglehub (pode demorar, ~2,5 GB)...")
    cache_path = Path(kagglehub.dataset_download(KAGGLE_DATASET_ID))
    print(f"Download concluido no cache do kagglehub: {cache_path}")

    dataset_dir_origem = encontrar_pasta_dataset(cache_path)

    RAW_DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)
    if RAW_DATASET_DIR.exists():
        print(f"Pasta de destino {RAW_DATASET_DIR} ja existe - removendo para copiar novamente.")
        shutil.rmtree(RAW_DATASET_DIR)

    print(f"Copiando dataset para {RAW_DATASET_DIR} (fora do cache do kagglehub)...")
    shutil.copytree(dataset_dir_origem, RAW_DATASET_DIR)

    data_yaml_path = RAW_DATASET_DIR / "data_local.yaml"
    gerar_data_yaml_local(RAW_DATASET_DIR, data_yaml_path)
    print(f"data.yaml local gerado em: {data_yaml_path}")

    for split in SPLITS:
        n_imgs = len(list((RAW_DATASET_DIR / split / "images").glob("*")))
        n_labels = len(list((RAW_DATASET_DIR / split / "labels").glob("*.txt")))
        print(f"  {split}: {n_imgs} imagens, {n_labels} arquivos de rotulo")

    print("\nEtapa 1a concluida. Proximo passo: python src/inspect_dataset.py")


if __name__ == "__main__":
    main()
