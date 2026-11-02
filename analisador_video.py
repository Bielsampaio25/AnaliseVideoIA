"""Orquestra detecção, rastreamento, eventos e geração dos artefatos."""
import csv
import json
from pathlib import Path
import cv2
from detector import Detector
from rastreador import RastreadorVideo, smooth_positions
from analise_jogadas import analisar_eventos
from config import RESULT_DIR

def analisar_video(caminho_video, modelo="yolo26n.pt", temporada="2026",
                   jogo_id="JOGO_001", jogador_id=None, metros_por_pixel=None):
    source = Path(caminho_video).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Vídeo não encontrado: {source}")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stem = source.stem
    out_video = RESULT_DIR / f"{stem}_analisado.mp4"
    out_csv = RESULT_DIR / f"{stem}_rastreamento.csv"
    out_json = RESULT_DIR / f"{stem}_metricas.json"
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Não foi possível criar o vídeo de saída.")

    detector = Detector(modelo)
    tracker = RastreadorVideo()
    rows = []
    frame_no = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            detections, yolo_result = detector.detectar(frame)
            tracker.update(frame_no, detections)
            annotated = yolo_result.plot()
            cv2.putText(annotated, f"Frame {frame_no}", (18, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            writer.write(annotated)
            for d in detections:
                rows.append({
                    "frame": frame_no, "time_s": round(frame_no/fps, 3),
                    "class_id": d["class_id"], "track_id": d["track_id"],
                    "confidence": round(d["confidence"], 4),
                    "x1": d["bbox"][0], "y1": d["bbox"][1],
                    "x2": d["bbox"][2], "y2": d["bbox"][3]
                })
    finally:
        cap.release()
        writer.release()

    selected = jogador_id if jogador_id is not None else tracker.jogador_mais_observado()
    duration = total_frames / fps if fps else 0
    tracks = tracker.players
    events = analisar_eventos(tracks, tracker.ball, fps, width, height)

    player_track = tracks.get(selected, []) if selected is not None else []
    observed_minutes = len(player_track) / fps / 60 if fps else 0

    distance_m = avg_speed = max_speed = None
    if metros_por_pixel and player_track:
        points = smooth_positions(player_track)
        speeds = []
        for a, b in zip(points, points[1:]):
            dt = (b["frame"]-a["frame"]) / fps
            if dt <= 0:
                continue
            dx, dy = (b["x"]-a["x"])*metros_por_pixel, (b["y"]-a["y"])*metros_por_pixel
            speed = ((dx*dx+dy*dy)**0.5) / dt * 3.6
            # Filtra saltos absurdos provocados por erro de rastreamento.
            if speed <= 40:
                speeds.append((speed, ((dx*dx+dy*dy)**0.5)))
        if speeds:
            distance_m = sum(x[1] for x in speeds)
            avg_speed = sum(x[0] for x in speeds)/len(speeds)
            max_speed = max(x[0] for x in speeds)

    metrics = {
        "jogador_id": f"TRACK_{selected}" if selected is not None else None,
        "temporada": temporada, "jogos": 1 if observed_minutes > 0 else 0,
        "minutos_jogados_observados": round(observed_minutes, 2),
        **events["event_metrics"],
        "distancia_total_km": round(distance_m/1000, 3) if distance_m is not None else None,
        "velocidade_media_kmh": round(avg_speed, 2) if avg_speed is not None else None,
        "velocidade_maxima_kmh": round(max_speed, 2) if max_speed is not None else None,
        "nota_media": None
    }
    report = {
        "source_video": str(source), "duration_s": round(duration, 2),
        "fps": fps, "resolution": f"{width}x{height}",
        "selected_track_id": selected, "ball_detections": len(
            [b for b in tracker.ball if not b.get("interpolated")]),
        "ball_trajectory": tracker.ball,
        "events": events["events"], "possession_samples": events["possession_samples"],
        "metricas": metrics, "limitations": events["limitations"],
        "files": {"video": str(out_video), "csv": str(out_csv), "json": str(out_json)}
    }
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=[
            "frame", "time_s", "class_id", "track_id", "confidence",
            "x1", "y1", "x2", "y2"
        ])
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    return report
