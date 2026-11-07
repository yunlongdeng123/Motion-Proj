"""J: native signed road-parent RGB operator and one bounded affinity check.

The fixed scene is H2 + C49 formal2K, without I. Only dummy feature leaves
receive backward. The deterministic colour perturbation is not a quality
candidate and performs no optimizer/model update.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np

B = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
R = B/'scene008_v44_a5_road_fusion_integration_20261107_r1'
NEW = B/'scene008_v44_a5_c49_rgb_20261107_r1'
SEM = B/'scene008_v44_full5_integration_20261107_r1'
REPO = Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
WORLD = B/'scene008_road_fusion_bake_20261107_r1/red_review/world_masks49_v1'
OBS = B/'scene008_instances_20261105_r1/observations/frames'
RAWS, STATES = [94, 105, 128, 137], ['keep', 'delete']
CONTRACT = R/'contract/J_parent_RGB_operator_v1.json'
SCENE = NEW/'formal_export2000_r1/scene_manifest.json'
QUAL = NEW/'parent_visibility49/qualification_manifest.json'
H2ROOT = R/'H4_model_foreground_culling_v2'
CACHE = NEW/'H2_material2000_retry1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(path):
    p = Path(path)
    return dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns, sha256=sha(p))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def signed(image, pair):
    # Keep the registered float32 residual/mean arithmetic used by evaluation.
    return (image[:, pair['plus_mask']].mean(1)
            - image[:, pair['minus_mask']].mean(1)).astype(np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=R/'J4_road_parent_operator')
    args = parser.parse_args()
    outdir, started = args.output, time.time()
    contract = json.loads(CONTRACT.read_text())
    assert contract['status'] == 'SEALED_BEFORE_J_GPU'
    assert contract['runtime_source_sha256'] == sha(__file__)
    assert contract['planned_operator_forwards'] == contract['planned_operator_backwards'] == 16
    assert contract['planned_perturbation_forwards'] == 8
    outdir.mkdir(exist_ok=False)
    (outdir/'batch_mass').mkdir()
    (outdir/'perturbation').mkdir()
    sys.path[:0] = [str(R/'source'), str(REPO),
        str(B/'scene008_v44_rgb_asset_20261107_r1/source')]
    from registered_seam_pairs import seam_pairs
    assert sha(R/'source/registered_seam_pairs.py') == '2f8598ccc0997e8a522316cddbb2290431253e5318c6cafc702c163c0a63b528'
    pairs = {raw: seam_pairs(raw, WORLD) for raw in RAWS}
    assert [len(pairs[raw]) for raw in RAWS] == [0, 14, 12, 27]
    qm = json.loads(QUAL.read_text())
    selection = QUAL.parent/qm['selection_file']
    assert sha(selection) == qm['selection_sha256']
    with np.load(selection, allow_pickle=False) as z:
        ids = z['eligible_parent_ids'].copy()
        assert np.array_equal(ids, np.flatnonzero(z['parent_mass'] > 0))
        eligible_mass = z['parent_mass'][ids].copy()
    assert ids.dtype == np.int64 and ids.shape == (2119,)
    scene_manifest = json.loads(SCENE.read_text())
    asset_path = SCENE.parent/scene_manifest['new_road']['file']
    assert sha(asset_path) == contract['material_asset_sha256']
    with np.load(asset_path, allow_pickle=False) as z:
        full_colours = z['parent_render_colours_srgb'].copy()
        mapping = z['child_to_parent_ids'].copy()
        assert np.array_equal(full_colours[mapping], z['render_colours_srgb'])
    assert full_colours.shape == (10098, 3) and full_colours.dtype == np.float32
    cache_records = json.loads((H2ROOT/'receipt.json').read_text())['records']
    input_paths = [CONTRACT, Path(__file__), QUAL, selection, SCENE, asset_path,
        H2ROOT/'receipt.json', R/'source/registered_seam_pairs.py',
        R/'source/foreground_road_culling_H4_v2.py', WORLD/'registry.json',
        WORLD/'primary_raw128/fixed_masks_raw128.npz']
    observed, baseline = {}, {}
    for raw in RAWS:
        p = OBS/f'{raw:03d}.npz'
        with np.load(p, allow_pickle=False) as z:
            O = z['rgb'].copy()
        assert O.shape == (320, 480, 3) and O.dtype == np.float32 and np.isfinite(O).all()
        observed[raw] = O.transpose(2, 0, 1)
        input_paths += [p, WORLD/f'{raw:03d}.npz']
        for state in STATES:
            p = CACHE/f'frames/{raw:03d}/{state}_float.npz'
            with np.load(p, allow_pickle=False) as z:
                baseline[(raw, state)] = {key: z[key].copy() for key in
                    ('rgb', 'alpha', 'depth_sum', 'depth_m', 'group_alpha_T')}
            input_paths += [p, H2ROOT/f'frames/{raw:03d}/{state}_visibility_mask.npz']
    inputs = {str(p): identity(p) for p in input_paths}
    write(outdir/'operator_protocol.json', dict(status='SEALED_BEFORE_J_NATIVE_CALLS',
        contract_sha256=sha(CONTRACT), inputs=inputs, parent_count=10098, variables=2119,
        pair_count_per_raw=[0, 14, 12, 27], source_ancestry_unchanged=True,
        feature_layout='original features0:16 + eight parent-only ones16:24 => extra14:22',
        residual_definition='predicted float32 RGB minus observed O, then original signed plus/minus means',
        full_parent_mass_closure='All10098 parents; selected2119 and frozen7979 are retained separately.',
        planned_operator_forwards=16, planned_operator_backwards=16,
        planned_perturbation_forwards=8, model_updates=0, quality_candidate=False))
    import torch
    from v44_runtime import V44Runtime
    from foreground_road_culling_H4_v2 import ForegroundRoadCulling
    from scripts.worldsim_v82.drivingeditor_dynamic_edit.v44_integration import SemanticActor5V44Integration
    from scripts.worldsim_v82.drivingeditor_dynamic_edit.road_fusion_integration import RoadFusionV44Integration
    from scripts.worldsim_v82.drivingeditor_dynamic_edit.baseline_road_appearance import baseline_appearance_binding
    assert sha(R/'source/foreground_road_culling_H4_v2.py') == '9eb7fb2f86af8a276d7b1503f119a6551d566b975b430f865af234aa55daabc8'
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    base = V44Runtime(B/'scene008_v44_rgb_asset_20261107_r1/contract/input_contract.json')
    sem = SemanticActor5V44Integration(base,
        semantic_sidecar_path=SEM/'semantic_actor5_calibration49/semantic_actor5_manifest.json',
        full5_transform_path=R/'contract/full5_transform_matrix.txt')
    road = RoadFusionV44Integration(sem, SCENE)
    h2 = ForegroundRoadCulling(road, max_iterations=64)
    for item in cache_records:
        raw, state = item['raw'], item['state']
        with np.load(H2ROOT/f'frames/{raw:03d}/{state}_visibility_mask.npz', allow_pickle=False) as z:
            mask, guard = z['selected_road_child_rows'], z['model_foreground_guard']
        record = item['meta']['foreground_road_culling']
        assert record['converged']
        h2.cache[(raw, state == 'delete')] = dict(mask=torch.as_tensor(mask, device='cuda'),
            guard=torch.as_tensor(guard, device='cuda'), record=record, baseline=None)
    model_binding = baseline_appearance_binding(sem)
    child_ids = torch.as_tensor(mapping, device='cuda', dtype=torch.long)
    full_vectors, selected_vectors, j0, pred0, obs0, records, batches = [], [], [], [], [], [], []
    calls = backs = 0

    def progress(phase, **fields):
        write(outdir/'kernel_progress.json', dict(phase=phase, actual_native_forwards=calls,
            actual_dummy_backwards=backs, optimizer_updates=0, **fields))

    def readout(native, scene):
        alpha = native['alpha'].detach().cpu().numpy()[0]
        depth = native['depth_sum'].detach().cpu().numpy()[0]
        return dict(rgb=(native['rgb']+(1-native['alpha'])*scene['env'].detach()).detach().cpu().numpy(),
            alpha=alpha, depth_sum=depth, depth_m=depth/np.maximum(alpha, 1e-8)/base.scale,
            group_alpha_T=native['extra'][:8].detach().cpu().numpy())

    for raw in RAWS:
        for state in STATES:
            if not pairs[raw]:
                continue
            scene = h2.scene(raw, delete=state == 'delete')
            original = {key: value.detach() if torch.is_tensor(value) else value
                        for key, value in scene['rows'].items()}
            offset = int(scene['meta']['latest_semantic_prefix_rows'])
            assert len(original['mask']) == offset+40392
            assert not bool((original['features'][:, 16:24] != 0).any())
            reference = baseline[(raw, state)]
            heads = []
            for pair in pairs[raw]:
                row = {key: value for key, value in pair.items() if key not in ('minus_mask', 'plus_mask')}
                row.update(state=state, operator_row=len(records))
                records.append(row)
                j0.append(signed(reference['rgb']-observed[raw], pair))
                pred0.append(signed(reference['rgb'], pair))
                obs0.append(signed(observed[raw], pair))
                heads.append((pair, row))
            for batch_index, begin in enumerate(range(0, len(heads), 8)):
                active_heads = heads[begin:begin+8]
                probe = torch.ones((10098, 8), device='cuda', dtype=torch.float32, requires_grad=True)
                feature = torch.cat((probe.new_zeros((offset, 8)), probe.index_select(0, child_ids)), 0)
                kw = dict(original, features=torch.cat((original['features'][:, :16], feature), 1))
                assert all(not value.requires_grad for key, value in kw.items()
                           if torch.is_tensor(value) and key != 'features')
                native = base.raster(scene['cam'], base.pipe, 0, kw, scene['bg'])
                calls += 1
                progress('operator', raw=raw, state=state, batch=batch_index)
                values = readout(native, scene)
                parity = {key: float(np.abs(value-reference[key]).max()) for key, value in values.items()}
                for key, value in values.items():
                    assert np.array_equal(value, reference[key]), (raw, state, batch_index, key, parity[key])
                read = native['extra'][14:22]
                ones_error = float((read-native['extra'][7:8]).abs().max().detach())
                assert ones_error <= 1e-6
                loss = read[0].sum()*0
                expected = []
                for channel, (pair, _) in enumerate(active_heads):
                    plus = torch.as_tensor(pair['plus_mask'], device='cuda')
                    minus = torch.as_tensor(pair['minus_mask'], device='cuda')
                    loss = loss+read[channel][plus].mean()-read[channel][minus].mean()
                    view = read[channel].detach().cpu().numpy().astype(np.float64)
                    expected.append(float(view[pair['plus_mask']].mean()-view[pair['minus_mask']].mean()))
                gradient = torch.autograd.grad(loss, probe)[0].detach().cpu().numpy()
                backs += 1
                progress('operator', raw=raw, state=state, batch=batch_index)
                assert gradient.shape == (10098, 8) and np.isfinite(gradient).all()
                closure = []
                for channel, (_, row) in enumerate(active_heads):
                    full = gradient[:, channel].astype(np.float64)
                    error = abs(float(full.sum())-expected[channel])
                    assert error <= 1e-6, (raw, state, batch_index, channel, error)
                    full_vectors.append((row['operator_row'], full))
                    selected_vectors.append((row['operator_row'], full[ids]))
                    closure.append(dict(operator_row=row['operator_row'], all10098_sum=float(full.sum()),
                        expected_signed_road_mass=expected[channel], absolute_error=error,
                        selected2119_sum=float(full[ids].sum()), frozen7979_sum=float(np.delete(full, ids).sum())))
                path = outdir/f'batch_mass/{raw:03d}_{state}_{batch_index:02d}.npz'
                np.savez_compressed(path, parent_signed_mass=gradient)
                batches.append(dict(raw=raw, state=state, batch=batch_index,
                    used_channels=len(active_heads), file=str(path.relative_to(outdir)), sha256=sha(path),
                    native_parity_max_abs=parity, ones_equals_newroad_alphaT_maxabs=ones_error,
                    signed_mass_closure=closure))
                write(outdir/'progress.json', dict(actual_native_forwards=calls, actual_dummy_backwards=backs,
                    operator_rows=len(selected_vectors), batches=batches))
                print(json.dumps(dict(phase='operator', raw=raw, state=state, batch=batch_index,
                    calls=calls, backwards=backs, rows=len(selected_vectors))), flush=True)
                del native, probe, feature, kw, read, loss, values
            del scene, original
    assert calls == backs == 16 and len(records) == len(j0) == len(full_vectors) == 106
    full_vectors.sort(key=lambda item: item[0])
    selected_vectors.sort(key=lambda item: item[0])
    A = np.stack([item[1] for item in selected_vectors])
    Afull = np.stack([item[1] for item in full_vectors])
    J0 = np.stack(j0)
    assert A.shape == (106, 2119) and np.array_equal(A, Afull[:, ids])
    road.assert_unchanged(full=True)
    assert model_binding == baseline_appearance_binding(sem)
    matrix = outdir/'road_parent_operator.npz'
    np.savez_compressed(matrix, eligible_parent_ids=ids, initial_parent_RGB=full_colours[ids],
        eligible_parent_mass=eligible_mass, full_parent_RGB=full_colours, A=A, full_parent_A=Afull, baseline_J=J0,
        baseline_signed_RGB=np.stack(pred0), observed_signed_RGB=np.stack(obs0))
    manifest = dict(schema='J-road-parent-rgb-signed-seam-operator-v1',
        matrix_file=matrix.name, matrix_sha256=sha(matrix), contract_sha256=sha(CONTRACT),
        material_scene_manifest_sha256=sha(SCENE), material_asset_sha256=sha(asset_path),
        qualification_sha256=sha(QUAL), selection_sha256=sha(selection),
        eligible_parent_mass_role='Exact original C49 fit-mask qualification mass; proxy only, not H2 visibility or a protection bound.',
        model_binding=model_binding, raw_ids=RAWS, states=STATES, seam_records=records,
        pair_count_per_raw_per_state=[0, 14, 12, 27], batches=batches,
        baseline_J_definition='signed means of float32(predicted RGB - observed O); includes frozen background and all frozen parents',
        observed_signed_RGB_role='Saved measured O constants, not zero and not omitted.',
        A_definition='signed native alphaT per selected parent; SUM four children on fixed H2 visibility',
        variable='delta = new selected parent RGB - initial_parent_RGB; box -c0 <= delta <= 1-c0',
        full_parent_A_role='All10098 mass closure, selected2119 + frozen7979. Only selected2119 may vary.',
        solver_not_run=True, quality_candidate=False, optimizer_updates=0)
    write(outdir/'operator_manifest.json', manifest)
    write(outdir/'operator_receipt.json', dict(status='PASS_J_ROAD_PARENT_SIGNED_OPERATOR_NO_FIT',
        actual_native_forwards=calls, actual_dummy_backwards=backs, optimizer_updates=0,
        rows=106, variables=2119, native_model_and_input_update_count=0,
        operator_manifest_sha256=sha(outdir/'operator_manifest.json')))

    # One deterministic, box-valid <=.001 perturbation; no parameter fitting.
    c0 = full_colours[ids].copy()
    signs = np.where((ids[:, None]*3+np.arange(3)) % 2 == 0, 1., -1.)
    c1 = np.clip(c0.astype(np.float64)+signs*.001, 0, 1).astype(np.float32)
    for _ in range(4):
        over = np.abs(c1.astype(np.float64)-c0.astype(np.float64)) > .001
        if not over.any():
            break
        c1[over] = np.nextafter(c1[over], c0[over])
    delta = c1.astype(np.float64)-c0.astype(np.float64)
    assert np.max(np.abs(delta)) <= .001 and c1.min() >= 0 and c1.max() <= 1
    new_parents = full_colours.copy()
    new_parents[ids] = c1
    child_rgb = torch.as_tensor(new_parents[mapping], device='cuda')
    predicted = J0+A@delta
    perturbation_rows = []
    for raw in RAWS:
        for state in STATES:
            scene = h2.scene(raw, delete=state == 'delete')
            kw = {key: value.detach() if torch.is_tensor(value) else value
                  for key, value in scene['rows'].items()}
            offset = int(scene['meta']['latest_semantic_prefix_rows'])
            kw['colors_precomp'] = torch.cat((kw['colors_precomp'][:offset], child_rgb), 0)
            features = kw['features'].clone()
            features[offset:, 13:16] = child_rgb
            kw['features'] = features
            with torch.no_grad():
                native = base.raster(scene['cam'], base.pipe, 0, kw, scene['bg'])
            calls += 1
            progress('perturbation', raw=raw, state=state)
            values = readout(native, scene)
            reference = baseline[(raw, state)]
            geometry = {}
            for key in ('alpha', 'depth_sum', 'depth_m', 'group_alpha_T'):
                geometry[key] = float(np.abs(values[key]-reference[key]).max())
                assert np.array_equal(values[key], reference[key]), (raw, state, key, geometry[key])
            checks = []
            for row in records:
                if row['raw'] != raw or row['state'] != state:
                    continue
                pair = pairs[raw][row['raw_pair_index']]
                actual = signed(values['rgb']-observed[raw], pair)
                estimate = predicted[row['operator_row']]
                error = float(np.max(np.abs(actual-estimate)))
                assert error <= 1e-6, (row['pair_id'], state, error)
                checks.append(dict(operator_row=row['operator_row'], actual_residual_J=actual.tolist(),
                    predicted_residual_J=estimate.tolist(), max_abs_error=error))
            path = outdir/f'perturbation/{raw:03d}_{state}_float.npz'
            np.savez_compressed(path, **values)
            perturbation_rows.append(dict(raw=raw, state=state, file=str(path.relative_to(outdir)),
                sha256=sha(path), geometry_maxabs=geometry, checks=checks,
                max_linearity_error=max([x['max_abs_error'] for x in checks] or [0.])))
            del native, scene, kw
    np.savez_compressed(outdir/'perturbation/colour_perturbation.npz', eligible_parent_ids=ids,
        initial_parent_RGB=c0, perturbed_parent_RGB=c1, actual_delta_RGB=delta)
    road.assert_unchanged(full=True)
    assert model_binding == baseline_appearance_binding(sem)
    for path, value in inputs.items():
        assert identity(path) == value, path
    assert calls == 24 and backs == 16 and h2.actual_dummy_backwards == 0
    result = dict(status='PASS_J_OPERATOR_AND_BOUNDED_NATIVE_AFFINITY_NO_FIT',
        actual_native_forwards=calls, actual_dummy_backwards=backs,
        operator_forwards=16, operator_backwards=16, perturbation_forwards=8,
        original_model_updates=0, material_optimizer_updates=0, quality_candidate=False,
        variables=2119, operator_rows=106, actual_perturbation_maxabs=float(np.abs(delta).max()),
        perturbation=perturbation_rows, max_native_linearity_error=max(x['max_linearity_error'] for x in perturbation_rows),
        input_bytes_mtime_and_model_binding_unchanged=True,
        operator_manifest_sha256=sha(outdir/'operator_manifest.json'),
        seconds=time.time()-started, peak_reserved_cuda_bytes=int(torch.cuda.max_memory_reserved()),
        automatic_solver=False, automatic_fit=False, automatic49=False, shutdown=False)
    write(outdir/'receipt.json', result)
    print(json.dumps({k: value for k, value in result.items() if k != 'perturbation'}), flush=True)


if __name__ == '__main__':
    main()
