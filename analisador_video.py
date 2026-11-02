"""
Analisa um vídeo de futebol e conecta os rastreamentos às métricas do atleta.

Uso:
    python analisador_video.py --video videos/video.mp4

Opcional:
    --jogador-id 1
    --goal-x 900
    --metros-por-pixel 0.05
    --modelo yolo26n.pt

Observações:
- IDs do ByteTrack são temporários, não identificam a pessoa real.
- Distância/velocidade em km só são calculadas se metros_por_pixel for informado.
- Aproximação da bola ao gol gera um CANDIDATO a finalização, não um gol confirmado.
- Dribles, passes, assistências e gols não são inventados: exigem análise contextual
  adicional e confirmação.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

from metricas_atleta import (
    EstatisticasPartida,
    calcular_estatisticas_temporada,
    calcular_velocidades_rastreamento,
)


def centro_caixa(x1, y1, x2, y2):
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def distancia(ponto_a, ponto_b):
    return math.hypot(ponto_a[0] - ponto_b[0], ponto_a[1] - ponto_b[1])


def analisar_video(
    caminho_video,
    jogador_id=None,
    goal_x=None,
    metros_por_pixel=None,
    modelo="yolo26n.pt",
    temporada="2026",
    jogo_id="JOGO_VIDEO_001",
):
    caminho_video = Path(caminho_video)
    if not caminho_video.exists():
        raise FileNotFoundError(f"Vídeo não encontrado: {caminho_video}")

    pasta_saida = Path("resultados")
    pasta_saida.mkdir(parents=True, exist_ok=True)

    captura = cv2.VideoCapture(str(caminho_video))
    if not captura.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {caminho_video}")

    fps = captura.get(cv2.CAP_PROP_FPS) or 25.0
    largura = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    duracao_segundos = total_frames / fps if fps > 0 else 0

    nome_base = caminho_video.stem
    caminho_video_saida = pasta_saida / f"{nome_base}_analisado.mp4"
    caminho_csv = pasta_saida / f"{nome_base}_rastreamento.csv"
    caminho_json = pasta_saida / f"{nome_base}_metricas.json"

    codec = cv2.VideoWriter_fourcc(*"mp4v")
    escritor = cv2.VideoWriter(
        str(caminho_video_saida), codec, fps, (largura, altura)
    )
    if not escritor.isOpened():
        captura.release()
        raise RuntimeError("Não foi possível criar o vídeo de saída.")

    detector = YOLO(modelo)

    # Guarda posição central por ID e frame.
    posicoes_jogadores = defaultdict(list)
    posicoes_bola = []
    distancias_bola_jogador = []
    candidatos_finalizacao = []
    jogadores_frames = defaultdict(int)
    bola_frames = 0
    frame_numero = 0
    jogador_escolhido = int(jogador_id) if jogador_id is not None else None
    ultimo_centro_bola = None
    ultimo_distancia_gol = None

    with open(caminho_csv, "w", newline="", encoding="utf-8") as arquivo_csv:
        colunas = [
            "frame", "tempo_s", "tipo", "track_id",
            "centro_x", "centro_y", "x1", "y1", "x2", "y2",
            "distancia_bola_jogador_px",
        ]
        writer = csv.DictWriter(arquivo_csv, fieldnames=colunas)
        writer.writeheader()

        try:
            while True:
                sucesso, frame = captura.read()
                if not sucesso:
                    break

                frame_numero += 1
                tempo_s = frame_numero / fps

                resultados = detector.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[0, 32],
                    conf=0.15,
                    verbose=False,
                )
                resultado = resultados[0]
                pessoas = []
                bolas = []

                if resultado.boxes is not None and resultado.boxes.xyxy is not None:
                    caixas = resultado.boxes.xyxy.cpu().tolist()
                    classes = resultado.boxes.cls.cpu().tolist()
                    ids = (
                        resultado.boxes.id.int().cpu().tolist()
                        if resultado.boxes.id is not None else [None] * len(caixas)
                    )

                    for caixa, classe, track_id in zip(caixas, classes, ids):
                        x1, y1, x2, y2 = map(float, caixa)
                        centro = centro_caixa(x1, y1, x2, y2)
                        item = {
                            "track_id": track_id,
                            "centro": centro,
                            "bbox": (x1, y1, x2, y2),
                        }
                        if int(classe) == 0 and track_id is not None:
                            pessoas.append(item)
                            posicoes_jogadores[int(track_id)].append(
                                (frame_numero, centro[0], centro[1])
                            )
                            jogadores_frames[int(track_id)] += 1
                        elif int(classe) == 32:
                            bolas.append(item)

                if bolas:
                    # Se houver mais de uma caixa de bola, usa a de maior confiança
                    # disponível pelo detector (primeira detecção nesta versão).
                    bola = bolas[0]
                    bola_centro = bola["centro"]
                    posicoes_bola.append((frame_numero, *bola_centro))
                    bola_frames += 1
                else:
                    bola_centro = None

                # Seleciona automaticamente o jogador mais rastreado no vídeo.
                if jogador_escolhido is None and jogadores_frames:
                    jogador_escolhido = max(jogadores_frames, key=jogadores_frames.get)

                jogador_atual = next(
                    (p for p in pessoas if p["track_id"] == jogador_escolhido),
                    None,
                )

                distancia_atual = None
                if jogador_atual and bola_centro:
                    distancia_atual = distancia(jogador_atual["centro"], bola_centro)
                    distancias_bola_jogador.append({
                        "frame": frame_numero,
                        "tempo_s": round(tempo_s, 3),
                        "jogador_id": jogador_escolhido,
                        "distancia_px": round(distancia_atual, 2),
                    })

                # Grava as detecções em CSV para permitir auditoria posterior.
                for pessoa in pessoas:
                    x1, y1, x2, y2 = pessoa["bbox"]
                    cx, cy = pessoa["centro"]
                    d = (
                        distancia_atual
                        if pessoa["track_id"] == jogador_escolhido else None
                    )
                    writer.writerow({
                        "frame": frame_numero,
                        "tempo_s": round(tempo_s, 3),
                        "tipo": "jogador",
                        "track_id": pessoa["track_id"],
                        "centro_x": round(cx, 2),
                        "centro_y": round(cy, 2),
                        "x1": round(x1, 2), "y1": round(y1, 2),
                        "x2": round(x2, 2), "y2": round(y2, 2),
                        "distancia_bola_jogador_px": (
                            round(d, 2) if d is not None else ""
                        ),
                    })

                for bola in bolas:
                    x1, y1, x2, y2 = bola["bbox"]
                    cx, cy = bola["centro"]
                    writer.writerow({
                        "frame": frame_numero,
                        "tempo_s": round(tempo_s, 3),
                        "tipo": "bola",
                        "track_id": bola["track_id"],
                        "centro_x": round(cx, 2),
                        "centro_y": round(cy, 2),
                        "x1": round(x1, 2), "y1": round(y1, 2),
                        "x2": round(x2, 2), "y2": round(y2, 2),
                        "distancia_bola_jogador_px": "",
                    })

                # Detecta apenas aproximação horizontal da bola ao gol.
                # Isso é um indício visual, não confirmação de chute ou gol.
                if bola_centro is not None and goal_x is not None:
                    distancia_gol = abs(goal_x - bola_centro[0])
                    if ultimo_distancia_gol is not None:
                        aproximando = distancia_gol < ultimo_distancia_gol
                        deslocou = (
                            ultimo_centro_bola is not None
                            and distancia(bola_centro, ultimo_centro_bola) > 3
                        )
                        if aproximando and deslocou:
                            candidatos_finalizacao.append({
                                "frame": frame_numero,
                                "tempo_s": round(tempo_s, 3),
                                "distancia_bola_gol_px": round(distancia_gol, 2),
                                "status": "candidato_nao_confirmado",
                            })
                    ultimo_distancia_gol = distancia_gol
                    ultimo_centro_bola = bola_centro

                anotado = resultado.plot()
                if jogador_atual:
                    cv2.putText(
                        anotado,
                        f"Atleta selecionado: ID {jogador_escolhido}",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2,
                    )
                    if distancia_atual is not None:
                        cv2.putText(
                            anotado,
                            f"Distancia bola-atleta: {distancia_atual:.1f} px",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2,
                        )
                escritor.write(anotado)

        finally:
            captura.release()
            escritor.release()

    # Calcula métricas de deslocamento do atleta escolhido.
    posicoes = (
        posicoes_jogadores.get(jogador_escolhido, [])
        if jogador_escolhido is not None else []
    )
    velocidades = []
    if metros_por_pixel is not None and posicoes:
        velocidades = calcular_velocidades_rastreamento(
            posicoes, fps, metros_por_pixel
        )

    distancia_total_m = (
        sum(item["distancia_m"] for item in velocidades)
        if velocidades else None
    )
    velocidade_media = (
        sum(item["velocidade_kmh"] for item in velocidades) / len(velocidades)
        if velocidades else None
    )
    velocidade_maxima = (
        max(item["velocidade_kmh"] for item in velocidades)
        if velocidades else None
    )

    # Minutos são estimados pelo tempo em que o ID foi observado.
    # Isso não equivale ao tempo oficial de jogo/participação.
    frames_jogador = len(posicoes)
    minutos_observados = (
        frames_jogador / fps / 60 if fps > 0 else 0
    )

    partida = EstatisticasPartida(
        jogador_id=f"TRACK_{jogador_escolhido}" if jogador_escolhido is not None else "NAO_IDENTIFICADO",
        temporada=temporada,
        jogo_id=jogo_id,
        minutos_jogados=round(minutos_observados, 2),
        gols=None,
        assistencias=None,
        finalizacoes=None,
        dribles=None,
        passes_completos=None,
        passes_tentados=None,
        distancia_m=distancia_total_m,
        velocidade_media_kmh=velocidade_media,
        velocidade_maxima_kmh=velocidade_maxima,
    )

    resumo = calcular_estatisticas_temporada(
        [partida], partida.jogador_id, temporada
    )

    resultado_final = {
        "video": str(caminho_video),
        "duracao_video_segundos": round(duracao_segundos, 2),
        "fps": fps,
        "resolucao": f"{largura}x{altura}",
        "jogador_track_id": jogador_escolhido,
        "frames_com_jogador": frames_jogador,
        "frames_com_bola": bola_frames,
        "minutos_observados_estimados": round(minutos_observados, 2),
        "distancias_bola_jogador": distancias_bola_jogador,
        "candidatos_finalizacao": candidatos_finalizacao,
        "escala_metros_por_pixel": metros_por_pixel,
        "metricas_temporada": resumo,
        "avisos": [
            "O ID do rastreador não representa a identidade real do atleta.",
            "Minutos observados não equivalem necessariamente aos minutos oficiais.",
            "Sem calibração, distância e velocidade reais ficam indisponíveis.",
            "Candidato a finalização não significa chute ou gol confirmado.",
            "Passe, drible, assistência e gol ainda exigem análise contextual.",
        ],
        "arquivos": {
            "video_anotado": str(caminho_video_saida),
            "rastreamento_csv": str(caminho_csv),
        },
    }

    caminho_json.write_text(
        json.dumps(resultado_final, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )

    return resultado_final, caminho_json


def main():
    parser = argparse.ArgumentParser(
        description="Conecta rastreamento YOLO/ByteTrack às métricas do atleta."
    )
    parser.add_argument("--video", required=True, help="Caminho do vídeo.")
    parser.add_argument("--jogador-id", type=int, default=None)
    parser.add_argument(
        "--goal-x", type=float, default=None,
        help="Coordenada X aproximada da linha do gol no vídeo.",
    )
    parser.add_argument(
        "--metros-por-pixel", type=float, default=None,
        help="Escala calibrada do campo; não informe um valor chutado.",
    )
    parser.add_argument("--modelo", default="yolo26n.pt")
    parser.add_argument("--temporada", default="2026")
    parser.add_argument("--jogo-id", default="JOGO_VIDEO_001")
    args = parser.parse_args()

    resultado, arquivo = analisar_video(
        caminho_video=args.video,
        jogador_id=args.jogador_id,
        goal_x=args.goal_x,
        metros_por_pixel=args.metros_por_pixel,
        modelo=args.modelo,
        temporada=args.temporada,
        jogo_id=args.jogo_id,
    )

    print("\nANÁLISE CONCLUÍDA")
    print(json.dumps(resultado["metricas_temporada"], ensure_ascii=False, indent=4))
    print(f"\nRelatório JSON: {arquivo}")
    print(f"Vídeo anotado: {resultado['arquivos']['video_anotado']}")
    print(f"Rastreamento CSV: {resultado['arquivos']['rastreamento_csv']}")


if __name__ == "__main__":
    main()
