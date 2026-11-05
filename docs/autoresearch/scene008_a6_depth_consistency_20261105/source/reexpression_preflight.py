"""BUILD-only whole-support evidence for later two-sided re-expression.

No row is excluded for having inherited clipping or zero physical-crop mass.
Points are true measured depths or independent PVD proposal depths, mapped
through the current native camera/pose; parent centers never supply points.
"""
import gc
import hashlib
import numpy as np
import torch

from consistency import RUN, BUILD, HELD, Consistency, Window, selected_hits, read, write

MASS_NAMES = ('front_full', 'front_crop', 'front_clipped', 'measured_full',
              'measured_clipped', 'front_crop_clipped')
CENTROID_NAMES = ('measured', 'proposal')
FRONT_MARGIN_M, MIN_ROLE_MASS, MIN_ROLE_FRAMES, MIN_SEPARATION_M = .12, .05, 2, .025


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_npz(path, **arrays):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def frame_roles(experiment, raw):
    """Return named whole/crop role masses, real-point sums and a receipt."""
    if raw not in BUILD:
        raise ValueError('Preflight processes BUILD frames only')
    window = experiment.frame(raw, 'black')
    trace, data = window.trace, experiment.tr.raw(raw)
    evidence = experiment.tr.evidence[raw]
    target, measured = evidence['mask'], data['depth_valid']
    assert not bool((target & measured).any()), 'Fixed target must exclude measured depths'
    reference = experiment.reference(raw)  # Independent of parent live/visibility.
    assert bool(torch.isfinite(reference[target]).all()) and bool((reference[target] > 0).all())
    assert bool(torch.isfinite(data['depth'][measured]).all()) and bool((data['depth'][measured] > 0).all())
    selected = selected_hits(window, target | measured)
    lineage = experiment.actor_lineage[raw]
    take = (selected['rows'] < len(lineage)) & (selected['weight'] > 0)
    hit, rows, pixel, slot = (selected[key][take] for key in ('hit', 'rows', 'pixel', 'slot'))
    weight = selected['weight'][take].double()
    owners = lineage[rows].long()
    count = len(experiment.tr.native_ids)
    assert bool(((owners >= 0) & (owners < count)).all())
    assert bool(torch.isfinite(weight).all()) and bool((weight >= 0).all())
    # f is an extra coverage ledger only; it never gates the whole roles.
    lookup = window.alpha.new_zeros(len(window.alpha))
    lookup[window.indices] = window.fraction
    fraction = lookup[hit].double()
    assert bool(torch.isfinite(fraction).all()) and bool((fraction >= 0).all())
    center = trace.depth[rows].double()/experiment.tr.scale
    assert bool(torch.isfinite(center).all())
    front = target.flatten()[pixel] & (center < reference.flatten()[pixel].double()-FRONT_MARGIN_M)
    observed = measured.flatten()[pixel]
    clipped = trace.kind[rows] == 1  # Primitive flag, not a UID-wide label.
    mass = weight.new_zeros((len(MASS_NAMES), count))
    fields = (weight*front, weight*front*fraction, weight*front*clipped,
              weight*observed, weight*observed*clipped, weight*front*fraction*clipped)
    for index, value in enumerate(fields):
        mass[index].index_add_(0, owners, value)
    point_sums = weight.new_zeros((2, count, 3))
    if len(rows):
        rays = Window.directions(trace)[slot]
        origin, direction = Window.in_car(experiment, trace, rays)
        for index, (role, depth) in enumerate(((observed, data['depth']), (front, reference))):
            z = depth.flatten()[pixel[role]].double()*experiment.tr.scale
            points = origin.double()+direction[role].double()*z[:, None]
            assert bool(torch.isfinite(points).all())
            point_sums[index].index_add_(0, owners[role], points*weight[role, None])
    value, xyz = mass.cpu().numpy(), point_sums.cpu().numpy()
    assert np.isfinite(value).all() and np.isfinite(xyz).all() and np.all(value >= 0)
    parent = trace.base()
    isolated = evidence['lower_alpha'][target].double()
    visible = parent['extra'][1][target].double()
    receipt = dict(raw=int(raw), target_pixels=int(target.sum()), measured_pixels=int(measured.sum()),
        native_positive_hits=int(len(rows)), front_hits=int(front.sum()), measured_hits=int(observed.sum()),
        front_hits_outside_crop=int((front & (fraction == 0)).sum()),
        masses=value.sum(axis=1).tolist(),
        isolated_lower_alpha=float(isolated.sum()), actual_lower_alphaT=float(visible.sum()),
        pvd_visibility_deficit=float((isolated-visible).clamp_min(0).sum()),
        center_feature_maxabs_m=(float((center-trace.feature[rows, 0].double()/experiment.tr.scale).abs().max()) if len(rows) else 0.))
    return value, xyz, receipt


def ratio(numerator, denominator):
    return float(numerator/denominator) if denominator > 0 else None


def role_summary(masses, mask):
    totals = masses[:, mask].sum(axis=1)
    return dict(rows=int(mask.sum()), masses=dict(zip(MASS_NAMES, totals.tolist())),
        front_crop_fraction=ratio(totals[1], totals[0]),
        front_clipped_fraction=ratio(totals[2], totals[0]),
        front_unclipped_fraction=ratio(totals[0]-totals[2], totals[0]),
        measured_clipped_fraction=ratio(totals[4], totals[3]),
        measured_unclipped_fraction=ratio(totals[3]-totals[4], totals[3]),
        crop_front_clipped_fraction=ratio(totals[5], totals[1]))


def main():
    contract = read(RUN/'contract.json')
    baseline = RUN.parent/contract['source_parent']
    baseline_hash = sha(baseline)
    assert len(BUILD) == 40 and not set(BUILD).intersection(HELD)
    experiment = Consistency()
    try:
        native_ids = experiment.tr.native_ids.cpu().numpy().copy()
        count = len(native_ids)
        assert len(np.unique(native_ids)) == count
        masses = np.zeros((len(MASS_NAMES), count), np.float64)
        xyz_sums = np.zeros((2, count, 3), np.float64)
        counts = np.zeros((2, count), np.int16)  # front, measured; each distinct raw once.
        receipts = []
        for raw in BUILD:
            current, points, receipt = frame_roles(experiment, raw)
            assert current.shape == masses.shape and points.shape == xyz_sums.shape
            masses += current; xyz_sums += points
            counts[0] += current[0] > 0
            counts[1] += current[3] > 0
            receipts.append(receipt)
            print('REEXPRESSION_PREFLIGHT_FRAME', raw, receipt['masses'], receipt['pvd_visibility_deficit'], flush=True)
        denominators = np.stack((masses[3], masses[0]))
        centroids = np.divide(xyz_sums, denominators[:, :, None], out=np.zeros_like(xyz_sums),
                              where=denominators[:, :, None] > 0)
        have_both = (denominators > 0).all(axis=0)
        delta = centroids[0]-centroids[1]
        separation = np.linalg.norm(delta, axis=1)
        separation[~have_both] = 0.
        normal = np.divide(delta, separation[:, None], out=np.zeros_like(delta), where=separation[:, None] > 0)
        cut = np.einsum('ij,ij->i', normal, .5*(centroids[0]+centroids[1]))
        eligible = ((masses[0] >= MIN_ROLE_MASS) & (masses[3] >= MIN_ROLE_MASS)
                    & (counts >= MIN_ROLE_FRAMES).all(axis=0) & (separation >= MIN_SEPARATION_M))
        assert np.isfinite(normal).all() and np.isfinite(cut).all()
        assert experiment.tr.exp.unchanged() and sha(baseline) == baseline_hash
        save_npz(RUN/'preflight_ledger.npz', native_row_id=native_ids, masses=masses,
            mass_names=np.asarray(MASS_NAMES), counts=counts, count_names=np.asarray(['front_frames', 'measured_frames']),
            xyz_sums=xyz_sums, centroids=centroids, centroid_names=np.asarray(CENTROID_NAMES),
            valid_centroids=denominators > 0, separation_m=separation, normal=normal, cut=cut, eligible=eligible,
            build_frames=np.asarray(BUILD, np.int16), held_frames=np.asarray(HELD, np.int16),
            baseline_hash=np.asarray(baseline_hash), boxes=experiment.boxes.cpu().numpy(), model_unchanged=np.bool_(True))
        deficit = sum(r['pvd_visibility_deficit'] for r in receipts)
        isolated = sum(r['isolated_lower_alpha'] for r in receipts)
        visible = sum(r['actual_lower_alphaT'] for r in receipts)
        write(RUN/'preflight.json', dict(status='EXECUTION_PASS', stage='stage0_coverage_and_separator_evidence_only', state='black',
            build=list(BUILD), held=list(HELD), held_processed=False, selection_used_heldout=False,
            baseline=str(baseline), baseline_hash=baseline_hash, model_unchanged=True, training_updates=0,
            mass_names=list(MASS_NAMES), all_roles=role_summary(masses, np.ones(count, bool)),
            mixed_roles=role_summary(masses, have_both), eligible=role_summary(masses, eligible),
            eligible_front_coverage=ratio(masses[0, eligible].sum(), masses[0].sum()),
            eligible_crop_front_coverage=ratio(masses[1, eligible].sum(), masses[1].sum()),
            eligible_measured_coverage=ratio(masses[3, eligible].sum(), masses[3].sum()),
            pvd_visibility=dict(isolated_alpha=isolated, actual_alphaT=visible, positive_deficit=deficit,
                                deficit_fraction=ratio(deficit, isolated)), frame_receipts=receipts,
            eligibility=dict(front_mass=MIN_ROLE_MASS, measured_mass=MIN_ROLE_MASS,
                front_frames=MIN_ROLE_FRAMES, measured_frames=MIN_ROLE_FRAMES, separation_m=MIN_SEPARATION_M,
                front_margin_m=FRONT_MARGIN_M, clipping_or_image_quality_filter=False),
            plane='n=(mean_measured-mean_proposal)/norm; b=n dot midpoint; observed-facing side is n dot x >= b',
            points='Actual BUILD LiDAR depth or independent PVD proposal depth, mapped at true native time into car coordinates; no parent-center points',
            scope=['Whole role mass uses joint parent alpha*T without a physical-crop gate; crop is a separate coverage ledger.',
                'Clip shares use each current primitive kind==1; one UID may contain clipped and unclipped children.',
                'PVD visibility deficit sums positive pixel differences, not foreground alpha*T attribution; they are different quantities.',
                'The separator is contribution-associated evidence, not semantic identity, free-space truth or a fidelity guarantee.',
                'All mixed eligible UIDs are retained; no unclipped-only or image-quality subset selection.'],
            ledger_sha256=sha(RUN/'preflight_ledger.npz')))
        print('REEXPRESSION_PREFLIGHT_DONE', int(eligible.sum()), deficit, flush=True)
    finally:
        del experiment
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    torch.set_num_threads(1)
    with torch.no_grad():
        main()
