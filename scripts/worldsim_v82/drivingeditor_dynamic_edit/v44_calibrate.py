"""Freeze visible-mass ownership using explicit manifest frames and instance labels.

This assigns persistent rows once, before any edit query. RGB, geometry and
native parameters are never optimized here. The >0.5 rule is inherited.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np


def calibrate(manifest, output):
    import torch
    from PIL import Image
    from .v44_factory import from_manifest
    from .v44_manifest import SceneManifest
    m = SceneManifest.load(manifest)
    if m.ownership is not None or m.surfaces or m.road or m.static_gates:
        raise ValueError('Ownership calibration requires a native-only scene manifest')
    scene, _ = from_manifest(m)
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    actors = tuple(sorted(m.actor_to_track))
    if not actors:
        raise ValueError('No registered actors')
    ns, nd = len(scene.static.get_xyz), len(scene.dynamic.get_xyz)
    masses = torch.zeros((len(actors), ns+nd), device=scene.static.get_xyz.device)
    total = masses.new_zeros(ns+nd); records = []; started = time.monotonic()
    ops = scene._bindings()
    for frame, data in zip(m.frames, m.data['frames']):
        path = Path(data['instance_mask'])
        if not path.is_absolute(): path = m.base/path
        cam = scene.camera_provider(frame.key)
        rows = ops.merge_rows(ops.prepare_rasterization(cam, scene.static, scene.pipe, dynamic=False),
                              ops.prepare_rasterization(cam, scene.dynamic, scene.pipe, dynamic=True))
        if len(rows['means3D']) != ns+nd:
            raise ValueError('Native topology changed during ownership calibration')
        features = rows['features'].new_zeros((ns+nd, 24))
        features[:, :2] = rows['features'][:, :2]
        features.requires_grad_(True)
        pack = ops.raster(cam, scene.pipe, 0, dict(rows, features=features), features.new_zeros(3))
        tag = pack['extra'][0]
        labels_np = np.asarray(Image.open(path).resize((cam.image_width, cam.image_height), Image.Resampling.NEAREST))
        if labels_np.ndim != 2 or labels_np.dtype.kind not in 'iu':
            raise ValueError('Instance labels must be an integer H,W array')
        present = sorted(set(np.unique(labels_np).tolist())-{0})
        if not set(present) <= set(actors): raise ValueError('Unregistered instance labels')
        labels = torch.as_tensor(labels_np.astype(np.int64), device=features.device)
        total += torch.autograd.grad(tag.sum(), features, retain_graph=bool(present))[0][:, 2].detach()
        for i, actor in enumerate(present):
            grad = torch.autograd.grad((tag*(labels == actor)).sum(), features,
                                      retain_graph=i+1<len(present))[0][:, 2].detach()
            masses[actors.index(actor)] += grad
        records.append(dict(key=frame.key, instance_mask=str(path),
                            mask_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), present_actor_ids=present))
        print(json.dumps(dict(frame=frame.key, classes=len(present))), flush=True)
        del pack, tag, rows, features
    best, which = masses.max(0); ratio = best/total.clamp_min(1e-20)
    ids = torch.as_tensor(actors, device=masses.device)[which]
    ids = torch.where((total>0) & (ratio>.5), ids, torch.zeros_like(ids))
    cpu = lambda a:a.detach().cpu().numpy()
    np.savez_compressed(out/'semantic_ownership.npz', static_actor_ids=cpu(ids[:ns]),
        dynamic_actor_ids=cpu(ids[ns:]), static_ratio=cpu(ratio[:ns]), dynamic_ratio=cpu(ratio[ns:]),
        static_mass=cpu(total[:ns]), dynamic_mass=cpu(total[ns:]))
    scene.assert_persistent_unchanged()
    receipt = dict(status='PASS_FROZEN_SHARED_VISIBLE_MASS_OWNERSHIP', threshold=.5, strict_greater=True,
        frames=records, model=str(m.model), iteration=m.iteration, static_rows=ns, dynamic_rows=nd,
        counts={str(a):dict(static=int((ids[:ns]==a).sum()), dynamic=int((ids[ns:]==a).sum())) for a in actors},
        gaussian_parameter_updates=0, edit_query_used=False, seconds=time.monotonic()-started)
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();calibrate(a.manifest,a.output)

if __name__=='__main__':main()
