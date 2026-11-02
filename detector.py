"""Detecção de pessoas e bola com YOLO. O modelo pode ser trocado por pesos próprios."""
from ultralytics import YOLO
from config import DEFAULT_MODEL, CONFIDENCE, PERSON_CLASS, BALL_CLASS

class Detector:
    def __init__(self, modelo=DEFAULT_MODEL, confianca=CONFIDENCE):
        self.model = YOLO(str(modelo))
        self.confianca = confianca

    def detectar(self, frame):
        results = self.model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            classes=[PERSON_CLASS, BALL_CLASS],
            conf=self.confianca, verbose=False
        )
        result = results[0]
        detections = []
        if result.boxes is None:
            return detections, result

        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        ids = (boxes.id.int().cpu().numpy()
               if boxes.id is not None else [None] * len(xyxy))

        for box, cls, conf, track_id in zip(xyxy, classes, confs, ids):
            x1, y1, x2, y2 = map(float, box)
            detections.append({
                "class_id": int(cls), "confidence": float(conf),
                "track_id": int(track_id) if track_id is not None else None,
                "bbox": [x1, y1, x2, y2],
                "center": [(x1+x2)/2, (y1+y2)/2]
            })
        return detections, result
