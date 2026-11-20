"""mAP from exported per-image predictions, matching the Ultralytics validator.

Used for image-level bootstrap intervals, which need the metric recomputed on
resampled image sets. The matching rule and the AP integration are the ones of
`ultralytics` (`BaseValidator.match_predictions` and `metrics.ap_per_class`), so a
point estimate here agrees with `model.val` up to the difference between the
prediction and validation pipelines.

Inputs are the COCO-style files written by `train_detector.export_predictions`:
ground truth with `image_id`, `category_id` (1-based) and xywh `bbox`, and
predictions with the same fields plus `score`.
"""
from __future__ import annotations

import collections

import numpy as np

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def _xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def _iou(a, b):
    """IoU matrix between N and M xyxy boxes."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-7)


def match(pred_classes, true_classes, iou):
    """Ultralytics greedy matching: highest IoU first, one-to-one, same class, per threshold.

    Returns an N_pred x 10 boolean array of true positives.
    """
    correct = np.zeros((len(pred_classes), len(IOU_THRESHOLDS)), dtype=bool)
    if not len(pred_classes) or not len(true_classes):
        return correct
    same = np.asarray(true_classes)[:, None] == np.asarray(pred_classes)[None, :]
    iou = (iou.T * same)  # rows: ground truth, columns: predictions
    for i, threshold in enumerate(IOU_THRESHOLDS):
        hits = np.nonzero(iou >= threshold)
        pairs = np.array(hits).T
        if not pairs.shape[0]:
            continue
        if pairs.shape[0] > 1:
            order = iou[pairs[:, 0], pairs[:, 1]].argsort()[::-1]
            pairs = pairs[order]
            pairs = pairs[np.unique(pairs[:, 1], return_index=True)[1]]
            pairs = pairs[np.unique(pairs[:, 0], return_index=True)[1]]
        correct[pairs[:, 1].astype(int), i] = True
    return correct


def prepare(groundtruth, predictions):
    """Per-image arrays: {image_id: (gt_boxes, gt_classes, pred_boxes, pred_classes, scores)}."""
    gt = collections.defaultdict(lambda: ([], []))
    for a in groundtruth["annotations"]:
        gt[a["image_id"]][0].append(_xyxy(a["bbox"]))
        gt[a["image_id"]][1].append(a["category_id"] - 1)
    pred = collections.defaultdict(lambda: ([], [], []))
    for p in predictions:
        pred[p["image_id"]][0].append(_xyxy(p["bbox"]))
        pred[p["image_id"]][1].append(p["category_id"] - 1)
        pred[p["image_id"]][2].append(p["score"])
    per_image = {}
    for image in groundtruth["images"]:
        key = image["id"]
        g_boxes, g_cls = gt[key]
        p_boxes, p_cls, scores = pred[key]
        correct = match(p_cls, g_cls, _iou(p_boxes, g_boxes))
        per_image[key] = dict(tp=correct, conf=np.asarray(scores, float), pred_cls=np.asarray(p_cls, int),
                              target_cls=np.asarray(g_cls, int))
    return per_image


def map_from(per_image, image_ids, nc=2):
    """mAP50:95, mAP50 and per-class AP50:95 over the listed images (repeats allowed)."""
    from ultralytics.utils.metrics import ap_per_class

    tp = np.concatenate([per_image[i]["tp"] for i in image_ids]) if image_ids else np.zeros((0, 10), bool)
    conf = np.concatenate([per_image[i]["conf"] for i in image_ids])
    pred_cls = np.concatenate([per_image[i]["pred_cls"] for i in image_ids])
    target_cls = np.concatenate([per_image[i]["target_cls"] for i in image_ids])
    result = ap_per_class(tp, conf, pred_cls, target_cls, plot=False)
    ap, classes = result[5], result[6]
    per_class = {int(c): float(ap[k].mean()) for k, c in enumerate(classes)}
    # A class absent from the resample contributes nothing, as in the validator.
    return dict(map50_95=float(ap.mean()) if len(ap) else 0.0, map50=float(ap[:, 0].mean()) if len(ap) else 0.0,
                per_class=per_class)
