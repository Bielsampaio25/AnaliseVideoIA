"""Ponto de entrada: execute este arquivo para analisar um vídeo."""
import argparse
import json
from pathlib import Path
from analisador_video import analisar_video

def main():
    parser = argparse.ArgumentParser(description="Análise automática de vídeo de futebol.")
    parser.add_argument("--video", required=True, help="Caminho do vídeo.")
    parser.add_argument("--modelo", default="yolo26n.pt")
    parser.add_argument("--temporada", default="2026")
    parser.add_argument("--jogo-id", default="JOGO_001")
    parser.add_argument("--jogador-id", type=int, default=None,
                        help="ID ByteTrack opcional; se omitido, escolhe o mais observado.")
    parser.add_argument("--metros-por-pixel", type=float, default=None,
                        help="Use somente se houver calibração conhecida.")
    args = parser.parse_args()

    report = analisar_video(
        args.video, modelo=args.modelo, temporada=args.temporada,
        jogo_id=args.jogo_id, jogador_id=args.jogador_id,
        metros_por_pixel=args.metros_por_pixel
    )
    print("\nANÁLISE CONCLUÍDA")
    print(json.dumps(report["metricas"], ensure_ascii=False, indent=4))
    print(f"\nRelatório: {report['files']['json']}")
    print(f"Vídeo anotado: {report['files']['video']}")
    print(f"CSV: {report['files']['csv']}")

if __name__ == "__main__":
    main()
