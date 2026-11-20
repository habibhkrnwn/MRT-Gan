"""Image-only harvest decision layer on top of the ripeness detector.

Detector output (class-aware NMS only, conf 0.001) is turned into fruits, a calibrated probability
of "ripe", and one of three decisions:

    Panen       p_ripe >= tau_high
    Cek manual  tau_low < p_ripe < tau_high
    Belum       p_ripe <= tau_low

Ground truth is the ripe/unripe label of the UPT PATPH Puspa Lebo officers (category 1 = ripe,
2 = unripe, COCO xywh boxes). Nothing here reads images; it works on prediction lists.
"""
from __future__ import annotations

import collections
import hashlib
import math

RIPE, UNRIPE = 1, 2
DECISIONS = ('Panen', 'Cek manual', 'Belum')
EPS = 1e-6


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0.0, min(ax2, bx2) - max(a[0], b[0])), max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def fruits(predictions, iou_merge=0.5):
    """Cross-class grouping: one fruit per cluster of overlapping ripe/unripe boxes.

    Boxes are visited by descending score; a box joins the first cluster whose seed box overlaps
    it at IoU >= iou_merge. The seed (highest score) gives the fruit box.
    """
    clusters = []
    for p in sorted(predictions, key=lambda p: -p['score']):
        for c in clusters:
            if iou(c['bbox'], p['bbox']) >= iou_merge:
                key = 's_ripe' if p['category_id'] == RIPE else 's_unripe'
                c[key] = max(c[key], p['score'])
                break
        else:
            clusters.append(dict(bbox=p['bbox'], s_ripe=p['score'] if p['category_id'] == RIPE else 0.0,
                                 s_unripe=p['score'] if p['category_id'] == UNRIPE else 0.0))
    for c in clusters:
        c['confidence'] = max(c['s_ripe'], c['s_unripe'])
        c['margin'] = math.log(c['s_ripe'] + EPS) - math.log(c['s_unripe'] + EPS)
    return clusters


def by_image(predictions):
    out = collections.defaultdict(list)
    for p in predictions:
        out[p['image_id']].append(p)
    return out


def match(gt_boxes, clusters, iou_min=0.5):
    """Greedy one-to-one GT/fruit matching by IoU. Returns {gt index: cluster index}."""
    pairs = sorted(((iou(g['bbox'], c['bbox']), gi, ci) for gi, g in enumerate(gt_boxes)
                    for ci, c in enumerate(clusters)), reverse=True)
    used_g, used_c, out = set(), set(), {}
    for v, gi, ci in pairs:
        if v < iou_min or gi in used_g or ci in used_c:
            continue
        used_g.add(gi); used_c.add(ci); out[gi] = ci
    return out


class Platt:
    """p = sigmoid(a * margin + b), fitted with L2-regularised logistic regression."""

    def __init__(self, a=1.0, b=0.0):
        self.a, self.b = a, b

    def fit(self, margins, labels):
        from sklearn.linear_model import LogisticRegression
        if len(set(labels)) < 2:
            raise ValueError('calibration needs both classes')
        model = LogisticRegression(C=1.0).fit([[m] for m in margins], labels)
        self.a, self.b = float(model.coef_[0][0]), float(model.intercept_[0])
        return self

    def __call__(self, margin):
        z = self.a * margin + self.b
        return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def decide(p, tau_low, tau_high):
    if p >= tau_high:
        return 'Panen'
    if p <= tau_low:
        return 'Belum'
    return 'Cek manual'


def collect(images, gt, predictions, tau_det, iou_merge=0.5):
    """Per image: fruits above tau_det, and matches to GT. Returns list of dicts."""
    preds, gts = by_image(predictions), collections.defaultdict(list)
    for a in gt['annotations']:
        gts[a['image_id']].append(a)
    rows = []
    for image_id in images:
        clusters = [c for c in fruits(preds.get(image_id, []), iou_merge) if c['confidence'] >= tau_det]
        g = gts.get(image_id, [])
        rows.append(dict(image_id=image_id, clusters=clusters, gt=g, match=match(g, clusters)))
    return rows


def matched_pairs(rows):
    for r in rows:
        for gi, ci in r['match'].items():
            yield r['clusters'][ci], r['gt'][gi]['category_id'] == RIPE


def choose_tau_det(rows_all_clusters, grid=None):
    """Detection threshold with the best fruit-level F1. When several thresholds tie (typical on
    this near-saturated validation set), the median of the tied range is used so the operating
    point sits away from both the missed-fruit and the false-fruit edge."""
    grid = grid or [round(0.05 * i, 2) for i in range(1, 20)]
    scores = []
    for t in grid:
        tp = fp = fn = 0
        for r in rows_all_clusters:
            keep = [c for c in r['clusters'] if c['confidence'] >= t]
            m = match(r['gt'], keep)
            tp += len(m); fp += len(keep) - len(m); fn += len(r['gt']) - len(m)
        scores.append((2 * tp / (2 * tp + fp + fn) if tp else 0.0, t))
    best = max(f for f, _ in scores)
    tied = [t for f, t in scores if f == best]
    return tied[len(tied) // 2]


def choose_thresholds(pairs, target=0.98, floor_high=0.90, ceil_low=0.10, grid_high=None, grid_low=None):
    """tau_high: smallest t >= floor_high with precision(Panen) >= target; tau_low: largest
    t <= ceil_low with precision(Belum) >= target. The pre-specified floors keep a "Cek manual"
    band when validation separates the classes perfectly, which would otherwise push both
    thresholds to 0.5. Falls back to never deciding that side."""
    grid_high = grid_high or [round(floor_high + 0.01 * i, 2) for i in range(int(round((1 - floor_high) * 100)))]
    grid_low = grid_low or [round(0.01 * i, 2) for i in range(1, int(round(ceil_low * 100)) + 1)]
    tau_high, tau_low = 1.01, -0.01
    for t in grid_high:
        sel = [y for p, y in pairs if p >= t]
        if sel and sum(sel) / len(sel) >= target:
            tau_high = t
            break
    for t in reversed(grid_low):
        sel = [not y for p, y in pairs if p <= t]
        if sel and sum(sel) / len(sel) >= target:
            tau_low = t
            break
    return tau_low, tau_high


def evaluate(rows, prob, tau_low, tau_high, rule='calibrated'):
    """Fruit-level decision metrics against GT labels.

    rule: 'calibrated' (thresholds on prob(margin)), 'argmax' (Panen if s_ripe >= s_unripe),
    'omyon' (ripe winner with confidence >= 0.80 -> Panen, ripe below -> Cek manual, unripe -> Belum).
    """
    table = collections.Counter()
    reliability = []
    false_fruits = collections.Counter()
    for r in rows:
        decided = {}
        for ci, c in enumerate(r['clusters']):
            if rule == 'calibrated':
                p = prob(c['margin']); d = decide(p, tau_low, tau_high)
            elif rule == 'argmax':
                p = None; d = 'Panen' if c['s_ripe'] >= c['s_unripe'] else 'Belum'
            elif rule == 'omyon':
                p = None
                d = ('Panen' if c['s_ripe'] >= 0.80 else 'Cek manual') if c['s_ripe'] >= c['s_unripe'] else 'Belum'
            else:
                raise ValueError(rule)
            decided[ci] = (d, p)
        matched = set(r['match'].values())
        for ci, (d, _) in decided.items():
            if ci not in matched:
                false_fruits[d] += 1
        for gi, g in enumerate(r['gt']):
            truth = 'ripe' if g['category_id'] == RIPE else 'unripe'
            if gi in r['match']:
                d, p = decided[r['match'][gi]]
                table[(truth, d)] += 1
                if p is not None:
                    reliability.append((p, truth == 'ripe'))
            else:
                table[(truth, 'Tidak terdeteksi')] += 1
    n = {t: sum(v for (tt, _), v in table.items() if tt == t) for t in ('ripe', 'unripe')}
    decided_n = sum(table[(t, d)] for t in n for d in ('Panen', 'Belum'))
    correct = table[('ripe', 'Panen')] + table[('unripe', 'Belum')]
    panen = table[('ripe', 'Panen')] + table[('unripe', 'Panen')]
    total = n['ripe'] + n['unripe']
    return dict(
        n_ripe=n['ripe'], n_unripe=n['unripe'],
        confusion={f'{t}->{d}': table[(t, d)] for t in n for d in (*DECISIONS, 'Tidak terdeteksi')},
        premature=dict(k=table[('unripe', 'Panen')], n=n['unripe']),
        missed_ripe=dict(k=table[('ripe', 'Belum')] + table[('ripe', 'Tidak terdeteksi')], n=n['ripe']),
        coverage=decided_n / total if total else float('nan'),
        selective_accuracy=correct / decided_n if decided_n else float('nan'),
        panen_precision=table[('ripe', 'Panen')] / panen if panen else float('nan'),
        manual=table[('ripe', 'Cek manual')] + table[('unripe', 'Cek manual')],
        false_fruits=dict(false_fruits),
        ece=ece(reliability) if reliability else None)


def ece(pairs, bins=10):
    total, err = len(pairs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(p, y) for p, y in pairs if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if sel:
            err += len(sel) / total * abs(sum(p for p, _ in sel) / len(sel) - sum(y for _, y in sel) / len(sel))
    return err


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def split_groups(image_groups, image_classes):
    """Calibration / measurement halves of validation, as in texture_study_eval: groups ordered by
    sha256('2026:'+group), stratified by class composition, every other group to calibration."""
    groups = sorted(set(image_groups.values()), key=lambda g: hashlib.sha256(('2026:' + g).encode()).hexdigest())
    strata = collections.defaultdict(list)
    for g in groups:
        classes = tuple(sorted({image_classes[i] for i in image_groups if image_groups[i] == g}))
        strata[classes].append(g)
    cal_groups = {g for gs in strata.values() for g in gs[::2]}
    ids = sorted(image_groups)
    cal = [i for i in ids if image_groups[i] in cal_groups]
    return cal, [i for i in ids if i not in cal]
