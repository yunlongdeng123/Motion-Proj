"""Register BUILD-witnessed old support for frozen measured surface groups.

Inputs come from an explicitly selected prior coupled run. Outputs belong to a
different run. Registration is rendered-contribution evidence, not per-point
semantic identity truth; mixed old support remains flagged as mixed.
"""
import argparse
import gc
import hashlib
from itertools import product
import os
from pathlib import Path
import sys

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import torch

MIN_GROUP_MASS = .05
MIN_BUILD_FRAMES = 2
NORMAL_MARGIN_M = .08
BOX_PADDING_M = .28


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu(value):
    return value.detach().cpu().numpy()


def weighted_quantiles(value, weight, quantiles=(.05, .95)):
    good = np.isfinite(value) & np.isfinite(weight) & (weight > 0)
    value, weight = value[good], weight[good]
    if not len(value):
        return None
    order = np.argsort(value, kind="stable")
    cumulative = np.cumsum(weight[order], dtype=np.float64)
    indices = np.searchsorted(cumulative, np.asarray(quantiles) * cumulative[-1], side="left")
    return value[order[np.minimum(indices, len(order) - 1)]]


def conditional_modes(experiment, trace, rows, camera_rays, Window):
    """Use each Gaussian's exact conditional precision and half-space metadata.

    The unconstrained ray mode is mu=b/a. A retained half-space clamps that mode
    to its surviving interval, exactly as specified by the existing shader.
    Camera-z parameter t is in model units; Window.in_car converts its direction
    to object-local metres per model unit before constructing the 3-D point.
    """
    ray = camera_rays.double()
    q = trace.terms[rows].double()
    x, y = ray[:, 0], ray[:, 1]
    precision = (q[:, 0] * x*x + 2*q[:, 1] * x*y + 2*q[:, 2] * x
                 + q[:, 3] * y*y + 2*q[:, 4] * y + q[:, 5]).clamp_min(1e-12)
    mode = (q[:, 6] * x + q[:, 7] * y + q[:, 8]) / precision
    sigma = precision.rsqrt()
    meta = trace.feature[rows, 6:].double()
    slope = (ray * meta[:, 10:13]).sum(1)
    clipped = (trace.kind[rows] == 1) & (slope.abs() * sigma > 1e-10)
    threshold = meta[:, 13] / torch.where(clipped, slope, torch.ones_like(slope))
    mode = torch.where(clipped & (slope > 0), torch.maximum(mode, threshold), mode)
    mode = torch.where(clipped & (slope < 0), torch.minimum(mode, threshold), mode)
    origin, direction = Window.in_car(experiment, trace, camera_rays)
    point = origin.double() + direction.double() * mode[:, None]
    return (point - experiment.group_center.double()) @ experiment.group_basis.double()


def frame_witnesses(experiment, raw, group_mask, Window):
    bundle = experiment.bundle(raw, "black")
    trace, window = bundle.trace, bundle.window
    group = group_mask(experiment, raw)
    data = experiment.tr.raw(raw)
    protect = (data["obs"] | experiment.tr.protection[raw]["outside"]) & ~group
    counts = (trace.offsets[1:] - trace.offsets[:-1]).long()
    pixels = trace.pixels.long()
    slots = torch.where((group | protect).flatten()[pixels] & (counts > 0))[0]
    counts = counts[slots]
    n = int(counts.sum())
    rows_count = len(experiment.zero)
    group_mass = torch.zeros(rows_count, device="cuda", dtype=torch.float64)
    protected_mass = torch.zeros_like(group_mass)
    receipt = dict(raw=raw, group_pixels=int(group.sum()), protected_pixels=int(protect.sum()),
                   selected_rays=len(slots), selected_hits=n)
    if not n:
        return cpu(group_mass), cpu(protected_mass), None, receipt
    bounds = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    ray = torch.arange(len(slots), device="cuda").repeat_interleave(counts, output_size=n)
    hit = trace.offsets[slots[ray]].long() + torch.arange(n, device="cuda") - bounds[ray]
    rows = window.ids[hit].long()
    # All contributors remain in the prefix, including PVD; proxy alpha is
    # exactly zero. The native terminal contributor is skipped at T_after<1e-4.
    alpha = bundle.alphas(0., 0.)[hit].double()
    logs = torch.cat((alpha.new_zeros(1), torch.log1p(-alpha).cumsum(0)))
    before = (logs[:-1] - logs[bounds[ray]]).exp()
    live = before * (1 - alpha) >= .0001
    weight = torch.where(live, before * alpha, 0.)
    lineage = experiment.actor_lineage[raw]
    native = (rows < len(lineage)) & (weight > 0)
    on_group = group.flatten()[pixels[slots]][ray]
    group_hit, protected_hit = native & on_group, native & ~on_group
    group_ids = lineage[rows[group_hit]].long()
    protected_ids = lineage[rows[protected_hit]].long()
    group_mass.index_add_(0, group_ids, weight[group_hit])
    protected_mass.index_add_(0, protected_ids, weight[protected_hit])
    camera_rays = Window.directions(trace)[slots][ray[group_hit]]
    mode = conditional_modes(experiment, trace, rows[group_hit], camera_rays, Window)
    witnesses = dict(ids=cpu(group_ids), normal=cpu(mode[:, 2]), weight=cpu(weight[group_hit]))
    receipt.update(group_mass=float(group_mass.sum()), protected_mass=float(protected_mass.sum()),
                   native_group_hits=int(group_hit.sum()), native_protected_hits=int(protected_hit.sum()))
    return cpu(group_mass), cpu(protected_mass), witnesses, receipt


def register(experiment, ledger, rank, source, output, group_mask, Window, BUILD, HELD, write):
    size = len(experiment.zero)
    group_mass, protected_mass = np.zeros(size), np.zeros(size)
    group_frames, protected_frames = np.zeros(size, np.int16), np.zeros(size, np.int16)
    samples, receipts = [], []
    for raw in BUILD:
        gm, pm, witness, receipt = frame_witnesses(experiment, raw, group_mask, Window)
        group_mass += gm; protected_mass += pm
        group_frames += gm > 0; protected_frames += pm > 0
        if witness is not None:
            samples.append(witness)
        receipts.append(receipt)
    eligible = (group_mass >= MIN_GROUP_MASS) & (group_frames >= MIN_BUILD_FRAMES)
    total = group_mass + protected_mass
    fraction = np.divide(group_mass, total, out=np.zeros_like(total), where=total > 0)
    mixed = (group_mass > 0) & (protected_mass > 0)
    normal = np.concatenate([s["normal"] for s in samples]) if samples else np.empty(0)
    weights = np.concatenate([s["weight"] for s in samples]) if samples else np.empty(0)
    ids = np.concatenate([s["ids"] for s in samples]) if samples else np.empty(0, np.int64)
    accepted = eligible[ids] & np.isfinite(normal)
    quantiles = weighted_quantiles(normal[accepted], weights[accepted])
    bounds = np.asarray(ledger["bounds"]).copy()
    half = np.asarray(ledger["box"], dtype=np.float64) / 2 + BOX_PADDING_M
    corners = np.asarray(list(product((-1., 1.), repeat=3))) * half
    projected = (corners - ledger["center"]) @ ledger["basis"][:, 2]
    box_interval = np.array([projected.min(), projected.max()])
    status = "NO_ELIGIBLE_LINKS" if not eligible.any() else "NO_FINITE_MODES"
    if quantiles is not None:
        bounds[0, 2] = max(box_interval[0], min(bounds[0, 2], quantiles[0] - NORMAL_MARGIN_M))
        bounds[1, 2] = min(box_interval[1], max(bounds[1, 2], quantiles[1] + NORMAL_MARGIN_M))
        status = "REGISTERED"
    assert np.array_equal(bounds[:, :2], ledger["bounds"][:, :2])
    assert np.all(bounds[1] > bounds[0])
    result = dict(ledger, eligible=eligible, group_mass=group_mass, protected_mass=protected_mass,
                  group_fraction=fraction, mixed=mixed, group_frame_count=group_frames,
                  protected_frame_count=protected_frames, native_row_id=cpu(experiment.tr.native_ids),
                  eligible_original_native_ids=np.flatnonzero(eligible), prism_bounds=bounds)
    destination = output / f"registered_ledger_{rank}.npz"
    np.savez_compressed(destination, **result)
    provenance = dict(
        status=status, rank=rank, source_run=str(source), output_run=str(output),
        build=BUILD, held=HELD, selection_used_heldout=False,
        source_ledger_sha256=sha256(source / f"group_ledger_{rank}.npz"),
        registered_ledger_sha256=sha256(destination), original_native_rows=size,
        eligible_count=int(eligible.sum()), eligible_original_native_ids=np.flatnonzero(eligible).tolist(),
        eligible_group_mass=float(group_mass[eligible].sum()),
        eligible_protected_mass=float(protected_mass[eligible].sum()),
        eligible_mixed_count=int((eligible & mixed).sum()),
        eligible_group_fraction=fraction[eligible].tolist(),
        original_bounds=np.asarray(ledger["bounds"]).tolist(), prism_bounds=bounds.tolist(),
        conditional_mode_normal_q05_q95=None if quantiles is None else quantiles.tolist(),
        padded_full_actor_normal_interval=box_interval.tolist(),
        registered=dict(min_group_alpha_T=MIN_GROUP_MASS, min_distinct_build_frames=MIN_BUILD_FRAMES,
                        normal_quantiles=[.05, .95], normal_margin_m=NORMAL_MARGIN_M,
                        full_actor_box_padding_m=BOX_PADDING_M, retire=0., gamma=0.),
        contribution="Exact joint native alpha*T, FP64 selected-ray segment sums, native terminal cutoff",
        coordinates="Conditional ray mode from precision q, clamped by actual half-space metadata; tangents frozen",
        interpretation="Rendered linkage and mixed-support witnesses; no per-point semantic identity claim",
        frame_receipts=receipts, training_updates=0,
    )
    write(output / f"registered_group{rank}.json", provenance)
    print("REGISTERED_SUPPORT", rank, status, int(eligible.sum()), bounds.tolist(), flush=True)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Prior coupled run directory")
    parser.add_argument("--output", type=Path, required=True, help="New registration run directory")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve() / "source"))
    from coupled import RUN, BUILD, HELD, Coupled, group_mask, read, write
    from volume import Window
    source, output = RUN.resolve(), args.output.resolve()
    if source != args.source.resolve() or source == output:
        raise ValueError("Explicit source must match coupled.RUN and outputs must use a different run")
    output.mkdir(parents=True, exist_ok=True)
    count = read(source / "groups.json")["qualified_count"]
    paths = [source / "groups.json"] + [source / f"group_ledger_{r}.npz" for r in range(count)]
    hashes = {str(path): sha256(path) for path in paths}
    results = []
    for rank in range(count):
        with np.load(source / f"group_ledger_{rank}.npz", allow_pickle=False) as bank:
            ledger = {key: bank[key] for key in bank.files}
        experiment = Coupled(ledger)
        try:
            results.append(register(experiment, ledger, rank, source, output,
                                    group_mask, Window, BUILD, HELD, write))
            assert experiment.tr.exp.unchanged()
            assert all(sha256(Path(path)) == digest for path, digest in hashes.items())
        finally:
            del experiment
            gc.collect()
            torch.cuda.empty_cache()
    write(output / "registration_summary.json", dict(
        status="EXECUTION_PASS", groups=count, source_run=str(source),
        source_hashes=hashes, registered_groups=[r["rank"] for r in results if r["status"] == "REGISTERED"],
        selection_used_heldout=False, training_updates=0))


if __name__ == "__main__":
    torch.set_num_threads(1)
    with torch.no_grad():
        main()
