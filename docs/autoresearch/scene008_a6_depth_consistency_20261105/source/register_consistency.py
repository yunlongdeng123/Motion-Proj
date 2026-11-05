"""Register persistent native-row eligibility on the 40 frozen BUILD frames.

The four mass rows are full target / center-front / center-and-mode-front /
protected crop contribution.  They measure joint parent alpha*T*f, not semantic
purity or actual post-retirement contribution changes.  PVD depth is a frozen
proposal reference obtained independently of parent occlusion, not truth.
"""
import gc
import hashlib
import numpy as np
import torch

from consistency import RUN, BUILD, HELD, Consistency, Window, selected_hits, read, write

MASS_NAMES = ('full_target', 'center_front', 'dual_front', 'protected')


def save_npz(path, **arrays):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def frame_masses(experiment, raw, margin_m=.12):
    """Return four dense FP64 parameter-indexed masses and a small receipt."""
    if raw not in BUILD:
        raise ValueError('Registration may process BUILD frames only')
    window = experiment.frame(raw, 'black')
    trace = window.trace
    data = experiment.tr.raw(raw)
    target = experiment.tr.evidence[raw]['mask']
    protect = (data['obs'] | data['depth_valid'] | experiment.tr.protection[raw]['outside']) & ~target
    assert not bool((target & protect).any())
    # This reference uses only fixed lower-PVD rows, with no parent-active gate.
    reference = experiment.reference(raw)
    assert bool(torch.isfinite(reference[target]).all()) and bool((reference[target] > 0).all())
    selected = selected_hits(window, target | protect)
    fraction = window.alpha.new_zeros(len(window.alpha))
    fraction[window.indices] = window.fraction
    lineage = experiment.actor_lineage[raw]
    take = ((selected['rows'] < len(lineage)) & (fraction[selected['hit']] > 0)
            & (selected['weight'] > 0))
    hit, rows, pixel, slot = (selected[key][take] for key in ('hit', 'rows', 'pixel', 'slot'))
    weight = selected['weight'][take].double() * fraction[hit].double()
    owner = lineage[rows].long()
    assert bool(torch.isfinite(weight).all()) and bool((weight >= 0).all())
    assert bool(((owner >= 0) & (owner < len(experiment.tr.native_ids))).all())
    on_target = target.flatten()[pixel]
    on_protect = protect.flatten()[pixel]
    center_front = torch.zeros_like(on_target)
    dual_front = torch.zeros_like(on_target)
    target_hit = torch.where(on_target)[0]
    if len(target_hit):
        # Actual raster center is the sorting reference.  RowWindow separately
        # checks feature[0] agrees with it; feature depth does not define C.
        center = trace.depth[rows[target_hit]].double() / experiment.tr.scale
        rays = Window.directions(trace)[slot[target_hit]]
        mode = window.crop_modes(hit[target_hit], rows[target_hit], rays)
        cutoff = reference.flatten()[pixel[target_hit]].double() - margin_m
        assert bool(torch.isfinite(center).all()) and bool(torch.isfinite(mode).all())
        center_front[target_hit] = center < cutoff
        dual_front[target_hit] = (center < cutoff) & (mode < cutoff)
    mass = weight.new_zeros((4, len(experiment.tr.native_ids)))
    for index, mask in enumerate((on_target, center_front, dual_front, on_protect)):
        mass[index].index_add_(0, owner[mask], weight[mask])
    value = mass.cpu().numpy()
    assert np.isfinite(value).all() and np.all(value >= 0)
    assert np.all(value[2] <= value[1]+1e-9) and np.all(value[1] <= value[0]+1e-9)
    receipt = dict(raw=int(raw), target_pixels=int(target.sum()), protected_pixels=int(protect.sum()),
        joint_selected_hits=int(len(selected['hit'])), positive_native_crop_hits=int(len(hit)),
        target_crop_hits=int(on_target.sum()), center_front_hits=int(center_front.sum()),
        dual_front_hits=int(dual_front.sum()), masses=value.sum(axis=1).tolist(),
        center_feature_maxabs_m=float(window.center_feature_maxabs_m))
    return value, receipt


def qualify(masses, framesC, framesCM, parameters):
    full, center, dual, protected = masses
    center_fraction = np.divide(center, full, out=np.zeros_like(full), where=full > 0)
    dual_fraction = np.divide(dual, full, out=np.zeros_like(full), where=full > 0)
    protected_fraction = np.divide(protected, full+protected, out=np.zeros_like(full), where=full+protected > 0)
    common = ((full >= parameters['minimum_target_mass'])
              & (protected_fraction <= parameters['max_protected_fraction']))
    eligible_center = (common & (center_fraction >= parameters['minimum_target_front_fraction'])
                       & (framesC >= parameters['minimum_distinct_BUILD_frames']))
    eligible_dual = (eligible_center & (dual_fraction >= parameters['minimum_target_front_fraction'])
                     & (framesCM >= parameters['minimum_distinct_BUILD_frames']))
    assert not np.any(eligible_dual & ~eligible_center) and np.all(framesCM <= framesC)
    doseC, doseCM = float(full[eligible_center].sum()), float(full[eligible_dual].sum())
    if doseC == 0 or doseCM == 0:
        coefficient = 0.
        applicability = 'INAPPLICABLE_ZERO_QUALIFIED_DOSE'
        reason = 'NO_CENTER_QUALIFIED_DOSE' if doseC == 0 else 'NO_DUAL_QUALIFIED_DOSE'
    else:
        coefficient = .5*doseCM/doseC
        applicability = ('DEGENERATE_IDENTICAL_ELIGIBILITY' if np.array_equal(eligible_center, eligible_dual)
                         else 'APPLICABLE')
        reason = 'C_EQUALS_CM' if applicability != 'APPLICABLE' else 'DISTINCT_DOSE_MATCHED_COMPARISON'
    if not np.isfinite(coefficient) or not 0 <= coefficient <= .5:
        raise ValueError(f'Out-of-range matched coefficient: {coefficient}; C={doseC}, CM={doseCM}')
    return dict(eligible_center=eligible_center, eligible_dual=eligible_dual,
                match_coefficient=coefficient, dose_center=doseC, dose_dual=doseCM,
                applicability=applicability, reason=reason)


def distribution(value):
    value = np.asarray(value, dtype=np.float64)
    return dict(count=int(value.size), quantile_0_50_90_99_100=(
        np.quantile(value, [0, .5, .9, .99, 1]).tolist() if value.size else []))


def main():
    contract_path = RUN/'contract.json'
    contract = read(contract_path)
    parameters = contract['qualification']
    assert len(BUILD) == 40 and not set(BUILD).intersection(HELD)
    experiment = Consistency()
    try:
        native_ids = experiment.tr.native_ids.cpu().numpy().copy()
        count = len(native_ids)
        assert len(np.unique(native_ids)) == count
        masses = np.zeros((4, count), np.float64)
        max_frame_mass = np.zeros_like(masses)
        framesC, framesCM = np.zeros(count, np.int16), np.zeros(count, np.int16)
        folder = RUN/'registration_frames'; folder.mkdir(exist_ok=True)
        receipts = []
        for raw in BUILD:
            current, receipt = frame_masses(experiment, raw, parameters['front_margin_m'])
            assert current.shape == masses.shape
            ids = np.flatnonzero((current > 0).any(axis=0))
            save_npz(folder/f'{raw:03d}.npz', raw=np.int16(raw), ids=ids,
                     native_row_id=native_ids[ids], masses=current[:, ids], mass_names=np.asarray(MASS_NAMES))
            masses += current
            np.maximum(max_frame_mass, current, out=max_frame_mass)
            framesC += current[1] > 0
            framesCM += current[2] > 0
            receipts.append(receipt)
            print('CONSISTENCY_REGISTER_FRAME', raw, len(ids), receipt['masses'], flush=True)
        selection = qualify(masses, framesC, framesCM, parameters)
        assert experiment.tr.exp.unchanged()
        concentration = np.divide(max_frame_mass, masses, out=np.zeros_like(masses), where=masses > 0)
        save_npz(RUN/'ledger.npz', native_row_id=native_ids, masses=masses,
            mass_names=np.asarray(MASS_NAMES), framesC=framesC, framesCM=framesCM,
            max_frame_mass=max_frame_mass, eligible_center=selection['eligible_center'],
            eligible_dual=selection['eligible_dual'], match_coefficient=np.float64(selection['match_coefficient']),
            dose_center=np.float64(selection['dose_center']), dose_dual=np.float64(selection['dose_dual']),
            applicability=np.asarray(selection['applicability']), boxes=experiment.boxes.cpu().numpy(),
            build_frames=np.asarray(BUILD, np.int16), held_frames=np.asarray(HELD, np.int16),
            contract_sha256=np.asarray(hashlib.sha256(contract_path.read_bytes()).hexdigest()),
            front_margin_m=np.float64(parameters['front_margin_m']), model_unchanged=np.bool_(True))
        summaries = {}
        for name, mask in [('all', np.ones(count, bool)), ('center', selection['eligible_center']), ('dual', selection['eligible_dual'])]:
            summaries[name] = dict(rows=int(mask.sum()), total_mass=masses[:, mask].sum(axis=1).tolist(),
                center_front_frames=distribution(framesC[mask]), dual_front_frames=distribution(framesCM[mask]),
                max_frame_fraction={key:distribution(concentration[i, mask & (masses[i] > 0)]) for i, key in enumerate(MASS_NAMES)},
                target_non_center_mass=float((masses[0]-masses[1])[mask].sum()),
                target_non_dual_mass=float((masses[0]-masses[2])[mask].sum()))
        write(RUN/'registration.json', dict(status='EXECUTION_PASS', build=list(BUILD), held=list(HELD),
            held_processed=False, selection_used_heldout=False, original_native_rows=count,
            mass_names=list(MASS_NAMES), statistics=summaries, frame_receipts=receipts,
            dose_center=selection['dose_center'], dose_dual=selection['dose_dual'],
            match_coefficient=selection['match_coefficient'], applicability=selection['applicability'],
            reason=selection['reason'], qualification=parameters, model_unchanged=True, training_updates=0,
            ledger_sha256=hashlib.sha256((RUN/'ledger.npz').read_bytes()).hexdigest(),
            sparse_index='ids indexes ledger.native_row_id; sparse native_row_id stores the global packed checkpoint ID',
            reference=contract['reference'], operator_dose='Full target crop mass on eligible lineages, including off-class target hits; first-order parent dose only',
            limits=['At most 20% of target mass may fail the corresponding front criterion; not geometric or semantic purity.',
                    'Off-class target mass is reported as full-C/full-CM, not merged into the common protected gate.',
                    'Missing measurements are not free space; PVD depth is a proposal, not hidden truth.',
                    'Center and dual have independently counted positive front-mass BUILD frames.']))
        print('CONSISTENCY_REGISTERED', {key:selection[key] for key in ('applicability', 'reason', 'dose_center', 'dose_dual', 'match_coefficient')}, flush=True)
    finally:
        del experiment
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    torch.set_num_threads(1)
    with torch.no_grad():
        main()
