"""在scene065冻结输入和真实浮点渲染上运行四阶段验收；从不读取隐藏道路GT。"""
from pathlib import Path
import argparse
import json
import os
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from PIL import Image

REPO = Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
sys.path.insert(0, str(REPO / 'scripts/worldsim_v82'))
from drivingeditor_dynamic_edit.road_asset_quality import (
    audit_completion, audit_world_material, audit_atlas_sampling, audit_render, rgb_difference, save_artifacts,
)

CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
LEGACY = CAMPAIGN / 'scene065_closedloop_20261116_r1'
FINAL = LEGACY / 'handoff_coverage_r2'
MATERIAL = CAMPAIGN / 'scene065_material_loop_20261114_r1'
PILOT_KEYS = [86, 110, 134, 179, 198]


def read(path):
    return json.loads(Path(path).read_text())


def load(path, keys=None):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in (keys or archive.files)}


def image(path):
    return np.asarray(Image.open(path).convert('RGB'), np.float32) / 255


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def completion_stage(out):
    control = read(CAMPAIGN / 'road_mask_decoupling_20261116_r1/output_guard_probe/control_freeze.json')
    records, guards = [], {}
    for row in control['cases']:
        if row['id'] not in ('scene065_raw086', 'scene065_raw110'):
            continue
        roles = load(row['native_guard_roles']['path'])
        observed, decoded, composite = [image(row[key]['path']) for key in ('observation', 'source_raw_B', 'composite_guard')]
        result = audit_completion(observed, decoded, composite, roles['W_guard'], roles['K'], roles['H_B'])
        result['case'] = row['id']
        result['source_paths'] = {key: row[key]['path'] for key in ('observation', 'source_raw_B', 'composite_guard', 'native_guard_roles')}
        save_artifacts(out / row['id'], result, rgb=composite)
        records.append(result)
        guards[int(row['id'][-3:])] = roles['W_guard']
    return dict(status='pass' if all(r['status'] == 'pass' for r in records) and len(records) == 2 else 'failed',
                cases=records, visual_quality_status='incomplete', human_verdict=None), guards


def material_stage(out, road_path, provenance_path, atlas_path, guards):
    cfg = read(MATERIAL / 'config.json')
    road = load(road_path)
    inputs = load(MATERIAL / 'field_inputs/world_field_input.npz')
    geometry = load(cfg['geometry_reference'], ('real', 'reference_srgb', 'keys', 'world_xyz'))
    base = load(LEGACY / 'field/road_visible_guarded_r2.npz')
    measured = inputs['context_rgb'].copy()
    measured[geometry['real']] = geometry['reference_srgb'][geometry['real']]
    control = read(CAMPAIGN / 'road_mask_decoupling_20261116_r1/output_guard_probe/control_freeze.json')
    source_images = {int(row['id'][-3:]): image(row['source_raw_B']['path']) for row in control['cases']
                     if row['id'] in ('scene065_raw086', 'scene065_raw110')}
    if provenance_path:
        provenance = load(provenance_path)
        if 'world_keys' not in provenance:
            provenance['world_keys'] = geometry['keys']
        provenance_semantics = 'explicit new continuous-material provenance'
    else:
        legacy = load(LEGACY / 'field/road_provenance.npz')
        effective_new = legacy['new_completion'] & ~(road.get('minimum_model_depth', np.zeros(len(road['active']))) > 0)
        source = np.zeros(len(road['active']), np.uint8)
        source[geometry['real']] = 1
        source[effective_new] = 3
        provenance = dict(source=source, unknown=source == 0, legal_write=effective_new,
                          known=geometry['real'], confidence=np.where(source > 0, 1., 0.).astype(np.float32),
                          row_rgb=road['reference_srgb'], observed_reference_srgb=geometry['reference_srgb'],
                          source_raw=legacy['best_source_raw'], source_uv_native=legacy['source_uv_native'],
                          world_keys=geometry['keys'])
        provenance_semantics = 'legacy source0 means unverified legacy RGB provenance, not a black/empty material; only real-observed and effective frozen generated rows are independently identified'
    result = audit_world_material(road, provenance, base_road=base,
                                  atlas=load(atlas_path) if atlas_path else None,
                                  source_guards=guards,
                                  source_images=source_images, observed_reference_srgb=measured,
                                  observed_eligible=inputs['known'] | geometry['real'],
                                  coverage_domain=provenance.get('coverage_requested', provenance.get('requested', inputs['requested'])) & provenance.get('surface_supported', inputs['surface_supported']))
    result['provenance_semantics'] = provenance_semantics
    result['source_paths'] = dict(road=str(road_path), provenance=str(provenance_path) if provenance_path else str(LEGACY / 'field/road_provenance.npz'),
                                 observations=str(MATERIAL / 'field_inputs/world_field_input.npz'), atlas=str(atlas_path) if atlas_path else None)
    if atlas_path:
        result['continuous_sampling'] = audit_atlas_sampling(atlas_path, provenance)
        save_artifacts(out / 'continuous_sampling', result['continuous_sampling'])
        if result['continuous_sampling']['status'] != 'pass':
            result['status'] = 'failed'
        marking_path = Path(atlas_path).parent / 'markings.npz'
        if marking_path.exists():
            markings = load(marking_path)
            result['marking_structure'] = dict(status='incomplete', observed_white_candidates=int(markings['observed_white'].sum()),
                                               observed_yellow_candidates=int(markings['observed_yellow'].sum()),
                                               dash_phase_known_cells=int(markings['phase_known'].sum()),
                                               inferred_marking_cells=int(markings['inferred_markings'].sum()),
                                               note='Observed color candidates only; no separate rendered curve or dash structure.')
    if atlas_path is None:
        result['missing_evidence'].append('Legacy output has no independently queryable continuous world atlas.')
        if result['status'] == 'pass':
            result['status'] = 'incomplete'
    save_artifacts(out, result)
    return result, float(road['meter_to_model'])


def region_stats(data, domain):
    count = int(domain.sum())
    only, road, static, dynamic = [np.asarray(data[name]).squeeze() for name in
                                   ('road_only_alpha', 'road_alphaT', 'static_alphaT', 'dynamic_alphaT')]
    def mean(a):
        return float(a[domain].mean()) if count else None
    return dict(pixels=count, road_only_alpha_mean=mean(only), road_alphaT_mean=mean(road),
                static_alphaT_mean=mean(static), dynamic_alphaT_mean=mean(dynamic),
                road_support_gt0p9_pixels=int((domain & (only > .9)).sum()),
                road_gap_lt0p05_pixels=int((domain & (only < .05)).sum()),
                static_gt0p1_pixels=int((domain & (static > .1)).sum()),
                dynamic_gt0p1_pixels=int((domain & (dynamic > .1)).sum()))


def rendering_stage(out, render_dir, keys, scale, baseline_dir):
    rows = []
    old_manifest = read(FINAL / 'persistent/scene.json')
    for raw in keys:
        data = load(render_dir / f'{raw:03d}/arrays.npz')
        diagnostics = load(FINAL / f'contribution_diagnostics_20261116/float/{raw:03d}.npz')
        if 'road_alphaT' not in data:
            data.update({key: diagnostics[key] for key in ('road_alphaT', 'static_alphaT', 'dynamic_alphaT')})
        data.setdefault('total_alpha', data.get('alpha'))
        domain = diagnostics['domain']
        roles = load(FINAL / f'domains/{raw:03d}.npz')
        baseline = load(baseline_dir / f'{raw:03d}/arrays.npz', ('rgb',))['rgb']
        historical_path = MATERIAL / f'rendered/corrected_delete/frames/{raw:06d}/arrays.npz'
        historical = load(historical_path, ('rgb',))['rgb']
        depth = {}
        for branch in ('static', 'dynamic', 'road'):
            name = branch + '_depth_mass_model'
            if name in data:
                contribution = data[branch + '_alphaT'].squeeze()
                depth[branch] = np.divide(data[name].squeeze(), contribution, out=np.full_like(contribution, np.nan), where=contribution > 1e-6) / scale
        part_names = ('static_rgb', 'dynamic_rgb', 'road_rgb', 'environment_rgb')
        parts = {key: data[key] for key in part_names} if all(key in data for key in part_names) else None
        result, masks = audit_render(data['road_only_alpha'], data['road_alphaT'], data['static_alphaT'],
                                     data['dynamic_alphaT'], data['total_alpha'], domain,
                                     rgb=data['rgb'], road_only_rgb=data.get('road_only_rgb'), parts=parts,
                                     protected=roles['K'], baseline_rgb=baseline, branch_depth=depth)
        result['raw'] = raw
        result['domain_definition'] = 'frozen road_views.target & inside & ~retained_reveal_guard & ~retained_box_guard & ~K; unchanged from previous contribution diagnostics'
        result['regions'] = dict(target_full=region_stats(data, roles['target']),
                                 target_fit=region_stats(data, roles['target'] & roles['fit']),
                                 root_fit=region_stats(data, roles['fit'] & ~roles['K']))
        label_path = old_manifest['frames'][raw]['instance_mask']
        labels = np.asarray(Image.open(label_path).resize((domain.shape[1], domain.shape[0]), Image.Resampling.NEAREST))
        result['protected_actors'] = {str(actor): rgb_difference(data['rgb'], baseline, labels == actor) for actor in (8, 21)}
        result['background'] = rgb_difference(data['rgb'], baseline, roles['protect'] & ~roles['K'])
        result['historical_protected'] = rgb_difference(data['rgb'], historical, roles['K'])
        result['historical_actors'] = {str(actor): rgb_difference(data['rgb'], historical, labels == actor) for actor in (8, 21)}
        result['source_paths'] = dict(arrays=str(render_dir / f'{raw:03d}/arrays.npz'),
                                    frozen_domain=str(FINAL / f'contribution_diagnostics_20261116/float/{raw:03d}.npz'),
                                    baseline=str(baseline_dir / f'{raw:03d}/arrays.npz'), historical=str(historical_path))
        if raw in PILOT_KEYS:
            save_artifacts(out / f'{raw:03d}', result, rgb=data['rgb'], masks=masks,
                           scalar_maps={name: data[name] for name in ('road_only_alpha', 'road_alphaT', 'static_alphaT', 'dynamic_alphaT')})
            if 'road_only_rgb' in data:
                save_artifacts(out / f'{raw:03d}/road_only', dict(source='actual float road_only_rgb', raw=raw), rgb=data['road_only_rgb'])
        rows.append(result)
        if raw % 25 == 0:
            print(json.dumps(dict(raw=raw, domain_pixels=result['metrics']['domain_pixels'], road_only=result['road_only_status'], joint=result['status'])), flush=True)
    total = sum(row['metrics']['domain_pixels'] for row in rows)
    summary = dict(frame_count=len(rows), road_domain_pixel_frames=total,
                   **{key: sum(row['metrics'][key] for row in rows) for key in
                      ('gap_pixels', 'thin_pixels', 'supported_pixels', 'supported_low_visible_pixels', 'static_dominated_pixels', 'dynamic_dominated_pixels')})
    summary['road_only_support_fraction'] = summary['supported_pixels'] / total if total else None
    for key in ('road_only_alpha_mean', 'road_alphaT_mean', 'static_alphaT_mean', 'dynamic_alphaT_mean'):
        summary[key] = sum(row['metrics'][key] * row['metrics']['domain_pixels'] for row in rows if row['metrics']['domain_pixels']) / total if total else None
    summary['protected'] = dict(pixel_frames=sum(row['metrics']['protected']['pixels'] for row in rows),
                                over_1byte_pixel_frames=sum(row['metrics']['protected']['over_tolerance_pixels'] for row in rows),
                                max_abs_rgb=max((row['metrics']['protected']['max_abs_rgb'] or 0) for row in rows))
    summary['conservation_max_error'] = max(row['metrics']['alphaT_conservation_max_error'] for row in rows)
    summary['branch_depth_frame_count'] = sum(row['metrics']['branch_depth'] is not None for row in rows)
    summary['principal_preservation_baseline'] = str(baseline_dir)
    summary['historical_protected'] = dict(pixel_frames=sum(row['historical_protected']['pixels'] for row in rows),
                                         over_1byte_pixel_frames=sum(row['historical_protected']['over_tolerance_pixels'] for row in rows),
                                         max_abs_rgb=max((row['historical_protected']['max_abs_rgb'] or 0) for row in rows))
    for region in ('target_full', 'target_fit', 'root_fit'):
        pixels = sum(row['regions'][region]['pixels'] for row in rows)
        summary[region] = dict(pixel_frames=pixels, **{name: sum(row['regions'][region][name] * row['regions'][region]['pixels'] for row in rows if row['regions'][region]['pixels']) / pixels if pixels else None
                                                     for name in ('road_only_alpha_mean', 'road_alphaT_mean', 'static_alphaT_mean', 'dynamic_alphaT_mean')})
    for actor in ('8', '21'):
        summary['protected_' + actor] = dict(pixel_frames=sum(row['protected_actors'][actor]['pixels'] for row in rows),
                                            over_1byte_pixel_frames=sum(row['protected_actors'][actor]['over_tolerance_pixels'] for row in rows),
                                            max_abs_rgb=max((row['protected_actors'][actor]['max_abs_rgb'] or 0) for row in rows))
    road_status = 'failed' if any(row['road_only_status'] == 'failed' for row in rows) else 'incomplete' if not total else 'pass'
    joint_status = 'failed' if any(row['status'] == 'failed' for row in rows) else 'incomplete'
    write(out / 'frames.json', rows)
    return dict(status=road_status, summary=summary, thresholds=rows[0]['road_only_thresholds']), dict(status=joint_status, summary=summary, visual_quality_status='incomplete', human_verdict=None, frames_file=str(out / 'frames.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--render-dir', type=Path, default=FINAL / 'rendered')
    parser.add_argument('--road', type=Path, default=LEGACY / 'field/road_visible_guarded_r2.npz')
    parser.add_argument('--material-provenance', type=Path)
    parser.add_argument('--atlas', type=Path)
    parser.add_argument('--baseline-render-dir', type=Path, default=FINAL / 'rendered')
    parser.add_argument('--material-only', action='store_true')
    parser.add_argument('--frames', default='all', help='all or comma-separated raw frame numbers')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    completion, guards = completion_stage(args.out / 'completion')
    material, scale = material_stage(args.out / 'material', args.road, args.material_provenance, args.atlas, guards)
    if args.material_only:
        write(args.out / 'acceptance.json', dict(stages=dict(completion=completion, world_material=material), rendering='not_run', human_verdict=None))
        print(json.dumps(dict(completion=completion['status'], world_material=material['status'], checks=material['checks'], metrics=material['metrics'])), flush=True)
        return
    keys = list(range(199)) if args.frames == 'all' else [int(v) for v in args.frames.split(',')]
    road, joint = rendering_stage(args.out / 'rendering', args.render_dir, keys, scale, args.baseline_render_dir)
    stages = dict(completion=completion, world_material=material, road_only=road, joint=joint)
    result = dict(schema='road_asset_four_stage_acceptance.v1',
                  scope='module2 acceptance; static/dynamic contributions are diagnostics, not current repair targets',
                  status='failed' if any(stage['status'] == 'failed' for stage in stages.values()) else 'incomplete',
                  stages=stages, frames=keys, full_199_frames=keys == list(range(199)),
                  human_verdict=None, visual_quality_status='incomplete', hidden_road_GT_used=False,
                  new_training_updates=0, new_GPU_forwards=0, seconds=time.time() - start,
                  script=str(Path(__file__).resolve()))
    write(args.out / 'acceptance.json', result)
    print(json.dumps(dict(status=result['status'], stages={key: value['status'] for key, value in stages.items()},
                          render_summary=joint['summary'], seconds=result['seconds']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
