"""
Teste rapido e isolado: valida que camera + YOLO + inferencia funcionam
NESTE computador, usando um modelo YOLO11n pre-treinado no COCO (baixado
automaticamente pelo Ultralytics). Nao depende do dataset de EPI.

Nao abre janela ao vivo (cv2.imshow pode nao renderizar bem quando o
processo e disparado por uma ferramenta/automacao). Em vez disso, captura
alguns frames da webcam, roda a deteccao e salva as imagens anotadas em
disco para inspecao visual.

Uso:
    python src/test_webcam_yolo.py [--camera 0] [--frames 5] [--modelo yolo11n.pt]
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ROOT_DIR

SAIDA_DIR = ROOT_DIR / "runs" / "teste_webcam"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="indice da webcam (padrao: 0)")
    parser.add_argument("--frames", type=int, default=5, help="quantos frames capturar e anotar")
    parser.add_argument("--modelo", type=str, default="yolo11n.pt", help="peso pre-treinado do Ultralytics")
    args = parser.parse_args()

    print(f"Carregando modelo '{args.modelo}' (baixa automaticamente se necessario)...")
    modelo = YOLO(args.modelo)

    dispositivo = "mps" if _mps_disponivel() else "cpu"
    print(f"Dispositivo de inferencia: {dispositivo}")

    print(f"Abrindo webcam indice {args.camera}...")
    captura = cv2.VideoCapture(args.camera)
    if not captura.isOpened():
        print(
            "PARADO: nao consegui abrir a webcam. No macOS, verifique se o Terminal/Python "
            "tem permissao de Camera em Ajustes do Sistema > Privacidade e Seguranca > Camera."
        )
        sys.exit(1)

    SAIDA_DIR.mkdir(parents=True, exist_ok=True)
    tempos_inferencia = []

    for i in range(args.frames):
        ok, frame = captura.read()
        if not ok:
            print(f"[aviso] falha ao ler frame {i}, pulando.")
            continue

        inicio = time.time()
        resultados = modelo.predict(frame, device=dispositivo, verbose=False)
        duracao_ms = (time.time() - inicio) * 1000
        tempos_inferencia.append(duracao_ms)

        resultado = resultados[0]
        frame_anotado = resultado.plot()
        caminho_saida = SAIDA_DIR / f"frame_{i:02d}.jpg"
        cv2.imwrite(str(caminho_saida), frame_anotado)

        classes_detectadas = [resultado.names[int(c)] for c in resultado.boxes.cls]
        print(f"frame {i}: {duracao_ms:.1f} ms | deteccoes: {classes_detectadas or 'nenhuma'} -> {caminho_saida}")

    captura.release()

    if tempos_inferencia:
        media_ms = sum(tempos_inferencia) / len(tempos_inferencia)
        print(f"\nTempo medio de inferencia: {media_ms:.1f} ms ({1000/media_ms:.1f} FPS teorico so de inferencia)")
    print(f"Frames anotados salvos em: {SAIDA_DIR}")
    print("\nPipeline camera + YOLO + inferencia OK neste computador.")


def _mps_disponivel() -> bool:
    import torch

    return torch.backends.mps.is_available()


if __name__ == "__main__":
    main()
