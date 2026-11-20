"""Automatic quality control of a CycleGAN bundle.

The six checks of the plan, in order:

1. target-domain classifier score  -- a small logistic model fitted on REAL train
   crops only; the candidate must reach `threshold` probability for its target class
2. background change ratio r_bg    -- measured on the uncomposited translation
3. colour distance to the target   -- standardised distance from phi(G(x)) to the
   target class centroid, both computed inside the mask
4. structural preservation         -- SSIM between parent crop and candidate,
   reported separately inside and outside the mask
5. duplicate filter                -- perceptual-hash clusters per parent, the same
   diagnostic `14_Benchmark_Kaggle/src/audit.py` already uses
6. blind visual audit              -- not automatic; `tools/review.py` does it

The rules produce a report and a recommendation. They never promote a candidate:
acceptance stays with the named reviewer, exactly as in the locked pipeline.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import config, partial
from .color import phi_numpy
from .config import CLASS_NAMES

BACKGROUND_REJECT_PERCENT = config.QC["reject_top_background_change_percent"]
CLASSIFIER_THRESHOLD = config.QC["target_classifier_threshold"]
DUPLICATE_THRESHOLD = 6


def _features(image, mask):
    config.benchmark_on_path()
    import audit

    return audit.lab_lbp(image, mask)


def fit_target_classifier(crops_dir, seed=0):
    """Logistic model on real train crops; returns (model, classes)."""
    from sklearn.linear_model import LogisticRegression

    crops_dir = Path(crops_dir)
    manifest = json.loads((crops_dir / "crops_manifest.json").read_text())
    features, labels = [], []
    for record in manifest["records"]:
        with Image.open(crops_dir / record["image"]) as handle:
            image = handle.convert("RGB")
        with Image.open(crops_dir / record["mask"]) as handle:
            mask = handle.convert("L")
        features.append(_features(image, mask))
        labels.append(int(record["class_id"]))
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    model.fit(np.stack(features), np.asarray(labels))
    return model, sorted(set(labels))


def ssim_numpy(a, b):
    """Global SSIM of two grayscale arrays in [0, 1], 8x8 uniform window."""
    from scipy.ndimage import uniform_filter

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a = uniform_filter(a, 7)
    mu_b = uniform_filter(b, 7)
    sigma_a = uniform_filter(a * a, 7) - mu_a * mu_a
    sigma_b = uniform_filter(b * b, 7) - mu_b * mu_b
    sigma_ab = uniform_filter(a * b, 7) - mu_a * mu_b
    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2)
    return numerator / denominator


def background_change_ratio(raw, source, mask):
    """r_bg = sum |(1-M) * (G(x)-x)| / sum |G(x)-x|, on the uncomposited output."""
    difference = np.abs(np.asarray(raw, np.float64) - np.asarray(source, np.float64))
    outside = (np.asarray(mask) == 0)[:, :, None]
    total = difference.sum()
    return float((difference * outside).sum() / (total + 1e-8))


def evaluate(bundle, crops_dir, dataset=None, threshold=CLASSIFIER_THRESHOLD, out=None,
             composite_trained=False):
    """Run every automatic check over a bundle and write `quality_control.json`.

    `composite_trained` marks a rung trained with the mask composite in the
    forward pass (G0X module M1). Such a generator gets no gradient outside the
    mask, so its raw output there is unconstrained and r_bg says nothing about
    it; r_bg is still reported, but it does not gate `recommended` and it is not
    comparable with rungs trained without the composite.
    """
    config.benchmark_on_path()
    import audit
    from common import Dataset

    bundle = Path(bundle)
    crops_dir = Path(crops_dir)
    dataset = dataset or Dataset(root=config.DATASET_ROOT)
    records = json.loads((bundle / "records.json").read_text())
    centroids = json.loads((crops_dir / "color_centroids.json").read_text())["classes"]
    centroid = {entry["class_id"]: (np.asarray(entry["centroid"]), np.asarray(entry["spread"]))
                for entry in centroids.values()}
    model, classes = fit_target_classifier(crops_dir)
    midpoint = partial.cr_midpoint(crops_dir / "color_centroids.json")

    rows, hashes, parents = [], {}, {}
    for record in records:
        with Image.open(bundle / record["image"]) as handle:
            candidate = handle.convert("RGB")
        with Image.open(bundle / record["mask"]) as handle:
            mask = handle.convert("L")
        source, parent_mask, _ = dataset.prepare(record["parent_id"], candidate.size[0])
        raw_path = bundle / record.get("raw", "")
        raw = candidate
        if record.get("raw") and raw_path.exists():
            with Image.open(raw_path) as handle:
                raw = handle.convert("RGB")

        target = int(record["class_id"])
        probability = float(model.predict_proba(_features(candidate, mask).reshape(1, -1))[0][classes.index(target)])
        selected = np.asarray(mask) > 0
        phi = phi_numpy(np.asarray(candidate, np.float64) / 255.0, selected)
        mean, spread = centroid[target]
        other = centroid[1 - target][0]
        distance = float(np.abs((phi - mean) / np.maximum(spread, 1e-6)).mean())
        distance_other = float(np.abs((phi - other) / np.maximum(spread, 1e-6)).mean())
        grey_source = np.asarray(source.convert("L"), np.float64) / 255.0
        grey_candidate = np.asarray(candidate.convert("L"), np.float64) / 255.0
        ssim_map = ssim_numpy(grey_source, grey_candidate)
        rows.append(dict(
            candidate_id=record["candidate_id"], parent_id=record["parent_id"],
            direction=record.get("direction", ""), target_class=CLASS_NAMES[target],
            classifier_probability=round(probability, 4),
            classifier_pass=bool(probability >= threshold),
            background_change_ratio=round(background_change_ratio(raw, source, parent_mask), 4),
            color_distance_target=round(distance, 4),
            color_distance_source=round(distance_other, 4),
            color_moves_to_target=bool(distance < distance_other),
            ssim_inside=round(float(ssim_map[selected].mean()), 4),
            ssim_outside=round(float(ssim_map[~selected].mean()), 4),
            partial_translation_share=round(partial.source_side_share(
                np.asarray(candidate, np.float64) / 255.0, selected, target, midpoint), 4),
        ))
        hashes[record["candidate_id"]] = audit.dhash(candidate)
        parents[record["candidate_id"]] = record["parent_id"]

    ratios = np.array([r["background_change_ratio"] for r in rows])
    cutoff = float(np.percentile(ratios, 100 - BACKGROUND_REJECT_PERCENT)) if len(ratios) else 1.0
    for row in rows:
        row["background_pass"] = True if composite_trained else bool(row["background_change_ratio"] <= cutoff)
        row["recommended"] = bool(row["classifier_pass"] and row["color_moves_to_target"] and row["background_pass"])

    diversity = audit.effective_variants(hashes, parents, DUPLICATE_THRESHOLD)
    report = dict(
        version=config.VERSION, bundle=str(bundle), crops=str(crops_dir),
        threshold=threshold, background_cutoff=None if composite_trained else round(cutoff, 4),
        composite_trained=bool(composite_trained),
        counts=dict(
            n_parent=len(set(parents.values())), n_generated=len(rows),
            n_recommended=int(sum(r["recommended"] for r in rows)),
            n_effective_clusters=diversity["total"],
        ),
        per_direction=dict(collections.Counter(r["direction"] for r in rows)),
        partial_translation=dict(
            rule="share of fruit pixels with Cr on the source side of the class midpoint",
            midpoint=round(midpoint, 6),
            mean_by_direction={d: round(float(np.mean([r["partial_translation_share"] for r in rows
                                                       if r["direction"] == d])), 4)
                               for d in sorted({r["direction"] for r in rows})},
        ),
        effective_diversity=diversity,
        rows=rows,
        note=("automatic checks only; acceptance still requires a named blind reviewer, "
              "and r_bg is measured on the uncomposited translation because compositing "
              "forces it to zero"
              + ("; this rung was trained with the mask composite, so r_bg on the raw "
                 "output is not comparable with other rungs and does not gate recommended"
                 if composite_trained else "")),
    )
    target = Path(out or bundle / "quality_control.json")
    target.write_text(json.dumps(report, indent=2) + "\n")
    return report
