"""Read-only source/data coverage audit. No CUDA, fit, query policy, or asset."""
from pathlib import Path
import hashlib
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

B = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
R = B/'scene008_v44_a5_road_fusion_integration_20261107_r1'
N = B/'scene008_v44_a5_c49_rgb_20261107_r1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def location(p, first, last):
    a = p.read_text().splitlines()
    return dict(path=str(p), sha256=sha(p), first_line=first, last_line=last,
        exact_lines=[dict(line=i+1, text=a[i]) for i in range(first-1, last)])


def main():
    qpath = N/'parent_visibility49/qualification_manifest.json'
    q = json.loads(qpath.read_text())
    selection = qpath.parent/q['selection_file']
    assert sha(selection) == q['selection_sha256']
    with np.load(selection) as z:
        mass, ids = z['parent_mass'], z['eligible_parent_ids']
        assert np.array_equal(ids, np.flatnonzero(mass > 0))
    qual_source = N/'source/qualify_C49_parent_visibility.py'
    core_source = N/'source/C49_material_core.py'
    qual_text = qual_source.read_text()
    core_text = core_source.read_text()
    assert "loss=view[torch.as_tensor(mask,device='cuda')].sum()" in qual_text
    assert 'eligible=np.flatnonzero(aggregate>0)' in qual_text
    assert "target=z['edgeblend_rgb'].copy();mask=z['fit_mask'].astype(bool)" in core_text
    assert ".5*(out['rgb'][:,mask]-target[:,mask]).square().mean()" in core_text
    assert 'protect_mask' not in core_text
    positive = []
    mass_inputs = []
    for raw in range(89, 138):
        for state in ['keep', 'delete']:
            path = qpath.parent/f'point_mass/{raw:03d}_{state}.npz'
            with np.load(path) as z:
                value = float(z['parent_mass'][13])
            mass_inputs.append(dict(path=str(path), sha256=sha(path)))
            if value > 0:
                positive.append(dict(raw=raw, state=state, legal_fit_alphaT_mass=value))
    assert sorted(set(a['raw'] for a in positive)) == [135, 136, 137]
    assert abs(sum(a['legal_fit_alphaT_mass'] for a in positive)-float(mass[13])) < 1e-15
    source_asset = B/'scene008_road_fusion_bake_20261107_r1/engineer/canonical_edgeblend_v1/new_road_canonical_candidate.npz'
    with np.load(source_asset) as z:
        valid = z['parent_source_valid'][:, 13]
        source_raw = z['parent_source_raw'][valid].tolist()
        source_rgb = z['parent_source_rgb'][valid, 13].tolist()
        original_source = dict(source_asset=str(source_asset), source_asset_sha256=sha(source_asset),
            parent_id=13, Cwarm_RGB=z['parent_render_colours_srgb'][13].tolist(),
            reference_RGB=z['parent_reference_srgb'][13].tolist(),
            valid_source_raw_ids=source_raw, valid_source_RGB=source_rgb,
            valid_source_count49=int(z['parent_source_valid_count49'][13]),
            role='Historical material source provenance only; not a query colour or visibility selector')
    assert source_raw == list(range(153, 161)) and original_source['valid_source_count49'] == 0
    target_records = []
    for raw, x, y in [(128, 301, 221), (137, 300, 227)]:
        target = next(a for a in q['targets'] if a['raw'] == raw)
        path = Path(target['path'])
        assert sha(path) == target['sha256']
        with np.load(path) as z:
            fit = z['fit_mask'].astype(bool)
            protect = z['protect_mask'].astype(bool)
            assert not (fit & protect).any() and not fit[y, x] and protect[y, x]
            target_records.append(dict(raw=raw, path=str(path), sha256=sha(path),
                current_role=target['current_role'], original_role=target['original_role'],
                pixel_xy=[x, y], fit_mask_at_pixel=bool(fit[y, x]), protect_mask_at_pixel=bool(protect[y, x]),
                edgeblend_RGB_at_pixel=z['edgeblend_rgb'][y, x].tolist(),
                observation_RGB_at_pixel=z['observation_rgb'][y, x].tolist(),
                fit_pixels=int(fit.sum()), protect_pixels=int(protect.sum()),
                fit_protect_overlap=0, loaded_target_key='edgeblend_rgb', loaded_loss_mask_key='fit_mask'))
    proof = json.loads((R/'white_point_parent_diagnostic_v1/receipt.json').read_text())
    parent = next(a for a in proof['records'][0]['parents_in_mass_order'] if a['parent_id'] == 13)
    report = dict(schema='C49_PROTECTED_SUPERVISION_COVERAGE_AUDIT_V1',
        status='PASS_READ_ONLY_SOURCE_AND_INPUT_PATH_AUDIT_COVERAGE_GAP_IDENTIFIED',
        architecture_reference='../../architecture_frozen21k_parallel.svg',
        qualification_source=[location(qual_source, 26, 34), location(qual_source, 50, 58), location(qual_source, 71, 79)],
        training_source=[location(core_source, 49, 64), location(core_source, 75, 82)],
        qualification_manifest=dict(path=str(qpath), sha256=sha(qpath)),
        selection=dict(path=str(selection), sha256=sha(selection), eligible_parent_count=len(ids)),
        actual_target_files=target_records,
        target_or_mask_wrong_file_read_detected=False,
        qualification_definition='SUM over all49 frames and KEEP/DELETE of joint parent-shared alphaT contribution inside the registered legal fit_mask; eligible iff aggregate > 0',
        qualification_has_protect_dominance_rejection=False,
        training_loss='Per sampled raw: sum over KEEP/DELETE of 0.5 * mean((native_rgb[:,fit_mask]-edgeblend_rgb[:,fit_mask])**2); only parent RGB is optimized',
        training_protect_mask_loss_present=False,
        training_scope='Complete C geometry on frozen21K scene; no H2 visibility mask during this C49 training. H2 is a later inference control.',
        parent13=dict(C49_fit_mass=float(mass[13]), positive_legal_fit_views=positive,
            colours=parent['colours'], original_source=original_source),
        measured_consequence='At the two registered protected pixels, parent13 contributes dominant visible mass and its RGB saturates from Cwarm about [0.391,0.425,0.471] to formal2K [0.990,1,1]. The pixels themselves have zero fit-mask loss weight.',
        interpretation='Positive aggregate fit mass establishes some supervision, not coverage of a parent\'s principal visible footprint. Reusing that parent colour at protected pixels is unconstrained by the executed colour loss.',
        qualification_tail_hypothesis='Small positive fit mass and strong protected-pixel visibility are observed; the precise spatial distribution of the saved fit mass is not available in these aggregates, so a tail-only explanation is not proved.',
        not_established=['Conflicting multi-view target RGB for this parent', 'Incorrect target reprojection', 'Wrong target or mask file loaded', 'Source-validity49=0 implies no rendered fit contribution'],
        source_provenance_boundary='The original direct source RGB exists at raw153..160 outside the current49 raw89..137; this alone proves neither wrong supervision nor multi-view conflict.',
        qualification_mass_files=mass_inputs,
        reference_probe=dict(path=str(R/'white_point_parent_diagnostic_v1/receipt.json'), sha256=sha(R/'white_point_parent_diagnostic_v1/receipt.json')),
        reference_counterfactual=dict(path=str(R/'review/positive_parent_union_Cwarm_counterfactual_cpu_v1.json'), sha256=sha(R/'review/positive_parent_union_Cwarm_counterfactual_cpu_v1.json')),
        GPU_forwards=0, backwards=0, optimizer_updates=0, solver_calls=0,
        query_policy_changed=False, assets_written=0, source_sha256=sha(__file__))
    out = R/'review/C49_protected_supervision_coverage_audit_v1.json'
    assert not out.exists()
    out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(out), sha256=sha(out), parent13_positive_views=positive,
        qualification_source_sha256=sha(qual_source), core_source_sha256=sha(core_source)), indent=2))


if __name__ == '__main__':
    main()
