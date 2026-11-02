"""Heurísticas conservadoras de eventos. Não afirma eventos sem evidência suficiente."""
import math
from collections import defaultdict
from rastreador import distance, smooth_positions

def analisar_eventos(player_tracks, ball_track, fps, width, height):
    events = []
    possession = []
    # Associa a bola ao jogador mais próximo, considerando a escala do corpo.
    by_frame = defaultdict(list)
    for pid, track in player_tracks.items():
        for p in track:
            by_frame[p["frame"]].append((pid, p))

    ball_by_frame = {b["frame"]: b for b in ball_track
                     if not b.get("interpolated", False)}
    for frame, ball in ball_by_frame.items():
        candidates = by_frame.get(frame, [])
        if not candidates:
            continue
        nearest = min(candidates, key=lambda item: distance(
            (item[1]["x"], item[1]["y"]), (ball["x"], ball["y"])))
        pid, p = nearest
        # Distância normalizada pela altura da caixa do jogador.
        box_h = max(30, p["bbox"][3] - p["bbox"][1])
        d = distance((p["x"], p["y"]), (ball["x"], ball["y"]))
        if d < box_h * 0.75:
            possession.append({"frame": frame, "player": pid, "distance": d})

    # Mudanças rápidas de posse podem indicar passe, mas não provam recepção.
    poss_by_frame = {p["frame"]: p["player"] for p in possession}
    frames = sorted(poss_by_frame)
    for a, b in zip(frames, frames[1:]):
        if b-a > max(3, int(fps*1.2)):
            continue
        if poss_by_frame[a] != poss_by_frame[b]:
            events.append({
                "type": "possible_pass", "frame": b,
                "from_track": poss_by_frame[a], "to_track": poss_by_frame[b],
                "confirmed": False
            })

    # Finalização candidata: aceleração/deslocamento forte da bola após posse.
    balls = smooth_positions(ball_track)
    for prev, cur in zip(balls, balls[1:]):
        dt = (cur["frame"] - prev["frame"]) / fps if fps > 0 else 0
        if dt <= 0 or cur.get("interpolated") or prev.get("interpolated"):
            continue
        speed_px_s = distance((prev["x"], prev["y"]), (cur["x"], cur["y"])) / dt
        if speed_px_s > max(width, height) * 0.8:
            events.append({
                "type": "possible_shot_or_long_ball", "frame": cur["frame"],
                "speed_px_s": round(speed_px_s, 2), "confirmed": False
            })

    # Drible/assistência/gol exigem contexto adicional; permanecem não avaliados.
    return {
        "possession_samples": possession,
        "events": events,
        "event_metrics": {
            "gols": None, "assistencias": None, "finalizacoes": None,
            "dribles": None, "passes_completos": None, "passes_tentados": None
        },
        "limitations": [
            "Mudança de posse é apenas candidata a passe, não confirmação.",
            "Velocidade da bola em pixels não confirma uma finalização.",
            "Gol exige localizar automaticamente a estrutura/linha do gol e confirmar a passagem completa da bola.",
            "O YOLO COCO padrão não detecta traves como classe própria."
        ]
    }
