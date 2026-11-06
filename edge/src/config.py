"""
Configuracoes centrais do projeto OpsSentinel.

Mantem em um unico lugar: identificador do dataset no Kaggle, mapa de
classes do dataset original e a lista de classes de interesse para este
MVP (deteccao de capacete e colete de seguranca).
"""

from pathlib import Path

# Diretorio raiz do projeto (pasta que contem este arquivo e "src/")
ROOT_DIR = Path(__file__).resolve().parent.parent

# Diretorios de dados e saidas (todos fora do controle de versao)
DATA_DIR = ROOT_DIR / "data"
RAW_DATASET_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT_DIR / "models"
RUNS_DIR = ROOT_DIR / "runs"

# Identificador do dataset no Kaggle (Roboflow v4, export YOLOv8, 640x640 stretch)
KAGGLE_DATASET_ID = "shlokraval/ppe-dataset-yolov8"

# Mapa de classes do data.yaml ORIGINAL do dataset (nc=14), por indice.
# Nao alteramos esses indices: o filtro de treino usa exatamente estes numeros.
ORIGINAL_CLASS_NAMES = {
    0: "Fall-Detected",
    1: "Gloves",
    2: "Goggles",
    3: "Hardhat",
    4: "Ladder",
    5: "Mask",
    6: "NO-Gloves",
    7: "NO-Goggles",
    8: "NO-Hardhat",
    9: "NO-Mask",
    10: "NO-Safety Vest",
    11: "Person",
    12: "Safety Cone",
    13: "Safety Vest",
}

# Classes de interesse para este MVP: Hardhat, NO-Hardhat, NO-Safety Vest,
# Person, Safety Vest. Passadas ao Ultralytics via parametro `classes=`.
TARGET_CLASS_IDS = [3, 8, 10, 11, 13]

# Classes negativas (ausencia de EPI) — usadas para priorizar recall e para
# a abordagem (a) de inferencia por classe negativa direta.
NEGATIVE_CLASS_IDS = [8, 10]  # NO-Hardhat, NO-Safety Vest

SPLITS = ["train", "valid", "test"]
