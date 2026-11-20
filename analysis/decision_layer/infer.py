"""Detector inference with the same settings as the training-time export (train_detector.py):
letterbox 640, conf 0.001, max_det 100, class-aware NMS only. Returns COCO-style predictions
with category 1 = ripe, 2 = unripe and image_id = file stem."""
from __future__ import annotations

import time
from pathlib import Path

IMAGE_SIZE, CONF, MAX_DET = 640, 0.001, 100


def load_model(weights):
    """The trainer saves {state_dict, cfg, nc, names}; rebuild the DetectionModel as the study
    evaluator does (texture_study_eval) and hang it on an Ultralytics wrapper."""
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    checkpoint = torch.load(str(weights), map_location='cpu', weights_only=False)
    net = DetectionModel(checkpoint['cfg'], nc=checkpoint['nc'], verbose=False)
    net.load_state_dict(checkpoint['state_dict'])
    net.names = dict(enumerate(checkpoint['names']))
    wrapper = YOLO(checkpoint['cfg'])
    wrapper.model = net.eval()
    return wrapper


def predict(model, items, device='cpu'):
    """items: iterable of paths, or of (image_id, path) pairs when file names differ from ids."""
    predictions, seconds = [], {}
    for item in items:
        image_id, path = (item if isinstance(item, tuple) else (Path(item).stem, item))
        path = Path(path)
        start = time.perf_counter()
        result = model.predict(source=str(path), imgsz=IMAGE_SIZE, conf=CONF, max_det=MAX_DET,
                               device=device, verbose=False)[0]
        seconds[image_id] = time.perf_counter() - start
        boxes = result.boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i]]
            predictions.append(dict(image_id=image_id, category_id=int(boxes.cls[i]) + 1,
                                    bbox=[round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                                    score=round(float(boxes.conf[i]), 6)))
    return predictions, seconds


def groundtruth(items):
    """COCO ground truth from YOLO txt labels (class 0 ripe, 1 unripe); items = (id, image, label)."""
    from PIL import Image
    images, annotations = [], []
    for image_id, path, label in items:
        with Image.open(path) as im:
            w, h = im.size
        images.append(dict(id=image_id, file_name=Path(path).name, width=w, height=h))
        for line in Path(label).read_text().splitlines():
            if not line.strip():
                continue
            c, x, y, bw, bh = map(float, line.split())
            annotations.append(dict(id=len(annotations) + 1, image_id=image_id, category_id=int(c) + 1,
                                    bbox=[(x - bw / 2) * w, (y - bh / 2) * h, bw * w, bh * h]))
    return dict(images=images, annotations=annotations, categories=[dict(id=1, name='ripe'), dict(id=2, name='unripe')])
