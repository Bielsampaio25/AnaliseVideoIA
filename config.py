from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIDEO_DIR = ROOT / "videos"
RESULT_DIR = ROOT / "resultados"
MODEL_DIR = ROOT / "models"

DEFAULT_MODEL = "yolo26n.pt"
# Opcional: modelo treinado para detectar trave/estrutura do gol.
# O modelo COCO padrão não possui uma classe confiável para "gol".
GOAL_MODEL = MODEL_DIR / "goal_detector.pt"

CONFIDENCE = 0.20
TRACKER = "bytetrack.yaml"
BALL_CLASS = 32
PERSON_CLASS = 0
MAX_BALL_GAP_FRAMES = 8
POSSESSION_DISTANCE_RATIO = 0.75
