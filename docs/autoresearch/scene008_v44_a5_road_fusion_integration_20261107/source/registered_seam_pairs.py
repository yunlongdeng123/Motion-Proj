"""Pure NumPy registered seam pairs, no environment writes/model/candidate reads.

Orientation is mean(plus residual) - mean(minus residual), exactly as in the
frozen evaluation. Order: each raw's world near_x, world lateral_y, then only
raw128 primary near_x. Each registry's existing bin order is retained.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

AXIS = {"near_x": 1, "lateral_y": 0}
EXPECTED_COUNTS = {94: 0, 105: 14, 128: 12, 137: 27}


def mask_sha256(mask):
    """SHA256 over C-order 320x480 uint8 0/1 bytes, without an array header."""
    return hashlib.sha256(np.ascontiguousarray(mask, dtype=np.uint8).tobytes()).hexdigest()


def _pairs(raw, scope, edge, minus, plus, xyz, registered_bins):
    minus = np.asarray(minus, bool)
    plus = np.asarray(plus, bool)
    assert minus.shape == plus.shape == (320, 480)
    selected = minus | plus
    bins = np.zeros(minus.shape, np.int32)
    bins[selected] = np.floor(np.asarray(xyz)[..., AXIS[edge]][selected]/.2).astype(np.int32)
    rows = []
    for pair in registered_bins:
        number = int(pair["tangent_bin"])
        left = minus & (bins == number)
        right = plus & (bins == number)
        nleft, nright = int(left.sum()), int(right.sum())
        assert nleft == pair["minus_count"] and nright == pair["plus_count"]
        assert nleft >= 3 and nright >= 3 and not (left & right).any()
        rows.append({"raw": int(raw), "scope": scope, "edge": edge,
                     "tangent_bin": number,
                     "pair_id": "raw%03d/%s/%s/bin%d" % (raw, scope, edge, number),
                     "minus": left, "plus": right,
                     "minus_count": nleft, "plus_count": nright,
                     "minus_sha256": mask_sha256(left), "plus_sha256": mask_sha256(right),
                     "orientation": "mean_plus_minus_mean_minus"})
    return rows


def primary_near_pairs(raw, masks):
    assert raw == 128
    minus = np.asarray(masks["band_near_x_minus"], bool)
    plus = np.asarray(masks["band_near_x_plus"], bool)
    selected = minus | plus
    bins = np.zeros(minus.shape, np.int32)
    bins[selected] = np.floor(masks["local_xyz"][...,1][selected]/.2).astype(np.int32)
    registered = []
    for number in np.unique(bins[selected]):
        left, right = minus & (bins == number), plus & (bins == number)
        if left.sum() >= 3 and right.sum() >= 3:
            registered.append({"tangent_bin": int(number), "minus_count":int(left.sum()), "plus_count":int(right.sum())})
    result = _pairs(raw, "primary128", "near_x", minus, plus, masks["local_xyz"], registered)
    assert len(result) == 6
    assert sum(r["minus_count"] for r in result) == 35
    assert sum(r["plus_count"] for r in result) == 33
    return result


def frame_pair_records(raw, world_masks, registered_edges, primary_masks=None):
    """Return actual bool masks plus SHA/count metadata for one original raw."""
    rows = []
    for edge in ("near_x", "lateral_y"):
        rows += _pairs(raw, "world", edge, world_masks[edge+"_minus"], world_masks[edge+"_plus"],
                       world_masks["local_xyz"], registered_edges[edge]["bins"])
    if raw == 128:
        if primary_masks is None:
            raise ValueError("Original independent primary128 mask is required")
        rows += primary_near_pairs(raw, primary_masks)
        world_near = [r for r in rows if r["scope"] == "world" and r["edge"] == "near_x"]
        assert sum(r["minus_count"] for r in world_near) == 36
        assert sum(r["plus_count"] for r in world_near) == 33
    assert len(rows) == EXPECTED_COUNTS[raw], (raw, len(rows))
    for index, row in enumerate(rows):
        row["raw_pair_index"] = index
    return rows


def pair_metadata(records):
    return [{k:v for k,v in row.items() if k not in ("minus", "plus")} for row in records]


def load_pilot_pair_records(world_root):
    """Read only the original registry/masks, returning all 53 original pairs."""
    world_root = Path(world_root)
    registry_path = world_root/"registry.json"
    expected = "0af33f63171cc34b404e4a0719495a16a49b993c5aaa36d1a442b1695251e7d8"
    assert hashlib.sha256(registry_path.read_bytes()).hexdigest() == expected
    registry = json.loads(registry_path.read_text())
    registered = {row["raw"]:row for row in registry["frames"]}
    primary_path=world_root/"primary_raw128/fixed_masks_raw128.npz"
    assert hashlib.sha256(primary_path.read_bytes()).hexdigest() == "53c560a34be8709d5b96267068aeabe68e76aa8a915fe2a97e93c67c8f439a06"
    with np.load(primary_path,allow_pickle=False) as z:
        primary = {k:z[k].copy() for k in z.files}
    result = []
    for raw in (94,105,128,137):
        path = world_root/("%03d.npz" % raw)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == registered[raw]["mask_sha256"]
        with np.load(path,allow_pickle=False) as z:
            world = {k:z[k].copy() for k in z.files}
        # Registry records store edges in `edges`, matching the original evaluator.
        result += frame_pair_records(raw,world,registered[raw]["edges"],primary if raw==128 else None)
    assert len(result) == 53
    for index,row in enumerate(result):
        row["pilot_pair_index"] = index
    return result


def seam_pairs(raw, world_root):
    """Small public worker ABI requested by G: original pair masks + metadata."""
    assert raw in EXPECTED_COUNTS
    records=[r for r in load_pilot_pair_records(world_root) if r['raw']==raw]
    return [{"raw":r["raw"],"edge":r["edge"],"kind":r["scope"],
             "tangent_bin":r["tangent_bin"],"minus_mask":r["minus"],"plus_mask":r["plus"],
             "minus_pixels":r["minus_count"],"plus_pixels":r["plus_count"],
             "minus_sha256":r["minus_sha256"],"plus_sha256":r["plus_sha256"],
             "pair_id":r["pair_id"],"raw_pair_index":r["raw_pair_index"],
             "pilot_pair_index":r["pilot_pair_index"],"orientation":r["orientation"]}
            for r in records]
