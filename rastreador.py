"""Histórico de trajetórias, suavização e pequenas lacunas de detecção."""
import math
from collections import defaultdict
import numpy as np
from config import PERSON_CLASS, BALL_CLASS, MAX_BALL_GAP_FRAMES

def center_bottom(box):
    x1, y1, x2, y2 = box
    return ((x1+x2)/2, y2)

class RastreadorVideo:
    def __init__(self):
        self.players = defaultdict(list)
        self.ball = []
        self.ball_last = None
        self.ball_gap = 0

    def update(self, frame_no, detections):
        for d in detections:
            if d["class_id"] == PERSON_CLASS and d["track_id"] is not None:
                x, y = center_bottom(d["bbox"])
                self.players[d["track_id"]].append(
                    {"frame": frame_no, "x": x, "y": y, "bbox": d["bbox"]}
                )
            elif d["class_id"] == BALL_CLASS:
                # Seleciona a detecção de maior confiança do frame.
                if self.ball_last is None or d["confidence"] > self.ball_last["confidence"]:
                    self.ball_last = {
                        "frame": frame_no, "x": d["center"][0],
                        "y": d["center"][1], "confidence": d["confidence"],
                        "bbox": d["bbox"]
                    }

        if self.ball_last and self.ball_last["frame"] == frame_no:
            self.ball.append(self.ball_last.copy())
            self.ball_gap = 0
        else:
            self.ball_gap += 1
            if self.ball and self.ball_gap <= MAX_BALL_GAP_FRAMES:
                # Interpolação apenas para continuidade visual; não conta como detecção observada.
                prev = self.ball[-1]
                self.ball.append({
                    "frame": frame_no, "x": prev["x"], "y": prev["y"],
                    "confidence": 0.0, "interpolated": True
                })
        self.ball_last = None

    def jogador_mais_observado(self):
        if not self.players:
            return None
        return max(self.players, key=lambda k: len(self.players[k]))

def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def smooth_positions(points, window=5):
    if len(points) < 3:
        return points
    result = []
    xs = np.array([p["x"] for p in points], dtype=float)
    ys = np.array([p["y"] for p in points], dtype=float)
    for i, p in enumerate(points):
        lo, hi = max(0, i-window//2), min(len(points), i+window//2+1)
        q = dict(p)
        q["x"] = float(np.median(xs[lo:hi]))
        q["y"] = float(np.median(ys[lo:hi]))
        result.append(q)
    return result
