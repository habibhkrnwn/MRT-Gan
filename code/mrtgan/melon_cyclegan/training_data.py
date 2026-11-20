"""Versioned GAN training pool and preflight. Never changes the frozen dataset.

The curated digest binds class, image AND mask bytes. The same pool is required
by G0-clean and G0X-v2. Original manifests/results remain readable and unchanged.
"""
from __future__ import annotations

import collections
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from . import config
from .color import phi_numpy


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(records):
    rows = [[r['id'], r['class_id'], r['domain'], r['image_sha256'],
             r['mask_sha256']] for r in sorted(records, key=lambda r: r['id'])]
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()


def verify(root, expected_policy=None):
    """Check the exact files the loader will see, not just the manifest header."""
    root = Path(root)
    manifest = json.loads((root / 'crops_manifest.json').read_text())
    if manifest.get('split') != 'train' or not manifest.get('records'):
        raise ValueError('training pool must be a nonempty train-only manifest')
    if expected_policy and manifest.get('training_policy') != expected_policy:
        raise ValueError('training data policy mismatch; prepare the curated pool first')
    records = manifest['records']
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('duplicate parent id in training manifest')
    for r in records:
        if r['split'] != 'train' or config.CLASS_DOMAIN[int(r['class_id'])] != r['domain']:
            raise ValueError(f"split/class/domain mismatch: {r['id']}")
        for key in ('image', 'mask'):
            path = (root / r[key]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError('crop path leaves pool')
            if sha(path) != r[f'{key}_sha256']:
                raise ValueError(f"{key} hash mismatch: {r['id']}")
    for domain in ('A', 'B'):
        for key, folder in [('image', f'train{domain}'), ('mask', f'train{domain}_mask')]:
            expected = {r[key] for r in records if r['domain'] == domain}
            actual = {str(p.relative_to(root)) for p in (root / folder).iterdir()
                      if p.suffix.lower() in ('.png', '.jpg', '.jpeg')}
            if not expected or actual != expected:
                raise ValueError(f'extra/missing files in {folder}')
    if digest(records) != manifest.get('crops_digest'):
        raise ValueError('curated crop digest mismatch')
    centroids = json.loads((root / 'color_centroids.json').read_text())
    if centroids.get('crops_digest') != manifest['crops_digest']:
        raise ValueError('colour centroids belong to another pool')
    if sha(root / 'color_centroids.json') != manifest.get('centroids_sha256'):
        raise ValueError('colour centroid file changed; recalibrate from the curated pool')
    return manifest


def curate(source, out, policy):
    source, out = Path(source), Path(out)
    if out.exists():
        raise ValueError('output already exists; use a new version or verify the existing pool')
    original = json.loads((source / 'crops_manifest.json').read_text())
    if original.get('split') != 'train' or original.get('crops_digest') != policy['source_crops_digest']:
        raise ValueError('source must be the pinned train crop pool')
    records, excluded, features = [], [], collections.defaultdict(list)
    for row in sorted(original['records'], key=lambda r: r['id']):
        if row['split'] != 'train':
            raise ValueError('non-train record in crop pool')
        reason = ('disputed_label' if row['id'] in policy['excluded_ids'] else
                  'fallback_mask' if row['mask_source'] not in policy['allowed_mask_sources'] else None)
        if reason:
            excluded.append(dict(id=row['id'], class_id=row['class_id'], reason=reason))
            continue
        record = dict(row)
        for key in ('image', 'mask'):
            path = (source / row[key]).resolve()
            if not path.is_relative_to(source.resolve()):
                raise ValueError('source path leaves pool')
            checksum = sha(path)
            if key == 'image' and checksum != row['image_sha256']:
                raise ValueError(f"source image changed: {row['id']}")
            record[f'{key}_sha256'] = checksum
        with Image.open(source / row['image']) as handle:
            rgb = np.asarray(handle.convert('RGB'), np.float64) / 255.0
        with Image.open(source / row['mask']) as handle:
            mask = np.asarray(handle.convert('L')) > 127
        features[row['class_id']].append(phi_numpy(rgb, mask))
        records.append(record)
    if set(features) != {config.RIPE, config.UNRIPE}:
        raise ValueError('curation emptied a domain')
    classes = {}
    for class_id, values in sorted(features.items()):
        x = np.stack(values)
        classes[config.CLASS_NAMES[class_id]] = dict(class_id=class_id, count=len(x),
                                                   centroid=x.mean(0).tolist(), spread=x.std(0).tolist())
    manifest = {k: v for k, v in original.items() if k not in ('records', 'counts', 'mask_sources', 'crops_digest')}
    manifest.update(records=records, counts=dict(collections.Counter(r['domain'] for r in records)),
                    mask_sources=dict(collections.Counter(r['mask_source'] for r in records)),
                    crops_digest=digest(records), training_policy=policy['name'],
                    curation_policy=policy, source_crops_digest=original['crops_digest'], excluded=excluded)
    out.mkdir(parents=True)
    for row in records:
        for key in ('image', 'mask'):
            dest = out / row[key]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / row[key], dest)
    (out / 'color_centroids.json').write_text(json.dumps(dict(
        version=config.VERSION, phi=original['phi'], source='curated train split real crops',
        crops_digest=manifest['crops_digest'], classes=classes), indent=2) + '\n')
    manifest['centroids_sha256'] = sha(out / 'color_centroids.json')
    (out / 'crops_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    verify(out, policy['name'])
    return manifest
