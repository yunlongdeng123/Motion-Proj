"""Construct a CPU-only native a5+a37 KEEP-only branch diagnostic; this is not a found historical run."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from pathlib import Path
import copy
import hashlib
import json
import shutil
import numpy as np
from PIL import Image

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a5_branch_control_20261107_r1')
OLD = Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/baseline_engineering_20261018/full5_pipeline/inputs/scene0008_089_137')
BASE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a37_only_B10k_20261103_r1')
ENGINE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/training_source/scene0008_089_137')
ACTORS = [(0, 0, 5, '3S_GJHyCGGK4ypFPVKOOqA', 21358),
          (1, 4, 37, 'vBOasuXpjI7h7d-X_VEQdA', 98452)]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ply_count(path):
    with path.open('rb') as handle:
        for _ in range(100):
            line = handle.readline().decode('ascii').strip()
            if line.startswith('element vertex '):
                return int(line.split()[-1])
            if line == 'end_header':
                break
    raise ValueError('No vertex count: ' + str(path))


def main():
    scene = ROOT / 'inputs/scene0008_089_137'
    scene.mkdir(parents=True, exist_ok=False)
    prep = json.loads((OLD / 'preparation_manifest.json').read_text())
    original_source = json.loads((OLD / 'dynamic_multi_actor_preparation.json').read_text())
    watched = [OLD / x for x in ['points3d.ply', 'transform_matrix.txt',
                                 'preparation_manifest.json', 'dynamic_multi_actor_preparation.json']]
    watched += [OLD / f'points3d_dynamic_{old_index}.ply' for _, old_index, _, _, _ in ACTORS]
    before = {str(p): {'bytes': p.stat().st_size, 'sha256': sha(p),
                       'mtime_ns': p.stat().st_mtime_ns} for p in watched}
    for name in ['calib', 'image', 'pose', 'sky', 'velodyne', 'annotations', 'timestamp', 'points3d.ply']:
        (scene / name).symlink_to(OLD / name, target_is_directory=(OLD / name).is_dir())
    # The native loader publishes this file; isolate that one writable artifact.
    shutil.copy2(OLD / 'transform_matrix.txt', scene / 'transform_matrix.txt')
    selected = []
    identity_rows = []
    for new_index, old_index, alias, track, expected_count in ACTORS:
        actor = copy.deepcopy(original_source['actors'][old_index])
        assert actor['actor_index'] == old_index and actor['raw_track_id'] == track
        source_ply = OLD / f'points3d_dynamic_{old_index}.ply'
        target_ply = scene / f'points3d_dynamic_{new_index}.ply'
        shutil.copy2(source_ply, target_ply)
        assert ply_count(target_ply) == actor['total_real_points'] == expected_count
        assert sha(source_ply) == sha(target_ply)
        source_asset = OLD / f'dynamic_actor_assets/actor_{old_index}'
        target_asset = scene / f'dynamic_actor_assets/actor_{new_index}'
        shutil.copytree(source_asset, target_asset)
        asset_hashes = {}
        for old_path in source_asset.rglob('*'):
            if old_path.is_file():
                new_path = target_asset / old_path.relative_to(source_asset)
                assert sha(old_path) == sha(new_path)
                asset_hashes[str(old_path.relative_to(source_asset))] = sha(new_path)
        actor.update(actor_index=new_index, source_actor_index=old_index, alias=alias)
        selected.append(actor)
        identity_rows.append(dict(new_actor_index=new_index, source_actor_index=old_index,
                                  alias=alias, raw_track_id=track, source_real_point_count=expected_count,
                                  source_ply=str(source_ply), target_ply=str(target_ply),
                                  ply_sha256=sha(target_ply), source_asset=str(source_asset),
                                  asset_file_sha256=asset_hashes))
    for name in ['dynamic_image/dynamic_image_0', 'dynamic_masks/dynamic_mask_0', 'mask/mask_0/Annotations']:
        (scene / name).mkdir(parents=True)
    frame_rows = []
    for frame in prep['frames']:
        fid = frame['id']
        rgb = np.asarray(Image.open(scene / 'image/image_0' / f'{fid}.png').convert('RGB'))
        bodies = [np.asarray(Image.open(scene / f'dynamic_actor_assets/actor_{new_index}/body_masks/{fid}.png')) > 0
                  for new_index, _, _, _, _ in ACTORS]
        assert all(body.shape == rgb.shape[:2] for body in bodies)
        union = np.logical_or.reduce(bodies)
        assert union.any()
        dynamic = np.where(union[:, :, None], rgb, 0).astype(np.uint8)
        for name in ['dynamic_masks/dynamic_mask_0', 'mask/mask_0/Annotations']:
            Image.fromarray(union.astype(np.uint8) * 255).save(scene / name / f'{fid}.png')
        Image.fromarray(dynamic).save(scene / f'dynamic_image/dynamic_image_0/{fid}.png')
        assert np.array_equal(np.asarray(Image.open(scene / f'dynamic_masks/dynamic_mask_0/{fid}.png')) > 0, union)
        assert np.array_equal(np.asarray(Image.open(scene / f'mask/mask_0/Annotations/{fid}.png')) > 0, union)
        assert np.array_equal(np.asarray(Image.open(scene / f'dynamic_image/dynamic_image_0/{fid}.png')), dynamic)
        # Validate both native resolution and the exact 480x320 supervision round trip.
        small_union = np.asarray(Image.fromarray(union.astype(np.uint8) * 255).resize((480, 320), Image.Resampling.NEAREST)) > 0
        source_small = [np.asarray(Image.open(OLD / f'dynamic_actor_assets/actor_{old_index}/body_masks/{fid}.png')
                                  .resize((480, 320), Image.Resampling.NEAREST)) > 0
                        for _, old_index, _, _, _ in ACTORS]
        assert np.array_equal(small_union, np.logical_or.reduce(source_small))
        frame_rows.append(dict(frame_id=fid, alias_pixels={str(row[2]): int(body.sum()) for row, body in zip(ACTORS, bodies)},
                               overlap_pixels=int(np.logical_and.reduce(bodies).sum()), union_pixels=int(union.sum()),
                               union_pixels480=int(small_union.sum()), union_exact=True, dynamic_RGB_exact=True,
                               mask_sha256=sha(scene / f'dynamic_masks/dynamic_mask_0/{fid}.png'),
                               dynamic_RGB_sha256=sha(scene / f'dynamic_image/dynamic_image_0/{fid}.png')))
    tracks = [row[3] for row in ACTORS]
    for key in ['selected_real_track_ids', 'selected_genuine_raw_track_ids', 'selected_raw_track_ids']:
        if key in prep:
            prep[key] = tracks
    prep['selected_real_track_ids'] = tracks
    prep['identity_source'] = dict(original_scene=str(OLD), actor_order=[5, 37],
                                   original_actor_indices=[0, 4], new_actor_indices=[0, 1],
                                   constructed_old_engine_subset=True, historical_run_found=False)
    write(scene / 'preparation_manifest.json', prep)
    source = copy.deepcopy(original_source)
    source.update(status='PASS_FIXED_A5_A37_KEEP_ONLY_INPUT', scene=str(scene), actors=selected,
                  actor_count=2, actor_order=[5, 37], source_actor_indices=[0, 4],
                  selection='Canonical full original actor_0=a5 and actor_4=a37 assets; constructed control, not found historical run',
                  historical_run_found=False, old_assets_modified=False, dynamic_RGB_and_union_exact=True)
    source.pop('selection_sha256', None)
    write(scene / 'dynamic_multi_actor_preparation.json', source)
    write(scene / 'dynamic_multi_actor_validation.json', source)
    config = json.loads((BASE / 'config.json').read_text())
    original_config = copy.deepcopy(config)
    config['source_path'] = str(scene)
    config['model_path'] = str(ROOT / 'formal5k')
    config['followup_scene_spec']['source_path'] = str(scene)
    config['followup_scene_spec']['actor_ply_paths'] = [str(scene / f'points3d_dynamic_{row[0]}.ply') for row in ACTORS]
    config.update(paper_eval_train_each_checkpoint=False, test_iterations=[], test_interval=1000000000,
                  save_iterations=[1000,5000], checkpoint_iterations=[1000,5000])
    assert config['iterations'] == 30000 and config['start_checkpoint'] is None
    assert config['dynamic_random_init_point'] == 0
    changes = {key: dict(parent=original_config[key], dual=config[key]) for key in original_config if original_config[key] != config[key]}
    assert set(changes) == {'source_path','model_path','followup_scene_spec','paper_eval_train_each_checkpoint','test_iterations','test_interval','save_iterations','checkpoint_iterations'}
    spec_changes = {key for key in original_config['followup_scene_spec']
                    if original_config['followup_scene_spec'][key] != config['followup_scene_spec'][key]}
    assert spec_changes == {'source_path', 'actor_ply_paths'}
    write(ROOT / 'config.json', config)
    after = {str(p): {'bytes': p.stat().st_size, 'sha256': sha(p),
                      'mtime_ns': p.stat().st_mtime_ns} for p in watched}
    assert before == after
    contract = dict(task_id='SCENE008-A5-BRANCH-CONTROL-20261107-R1',
                    purpose='Construct a full original PLY a5+a37 native dynamic subset for KEEP-only branch-responsibility diagnosis; not an a37-only deletion solution',
                    historical_run_found=False, constructed_old_engine_subset=True,
                    dynamic_actor_aliases=[5, 37], dynamic_raw_track_ids=tracks, source_actor_indices=[0, 4],
                    new_actor_indices=[0, 1], source_real_point_count=119810,
                    dynamic_source_real_points=[21358, 98452], actor_mapping=identity_rows,
                    retained_aliases=[26, 18, 6], retained_representation='a5 is deliberately dynamic for KEEP-only diagnosis; native render_s deletes a5+a37 and is not a preservation result',
                    engine=str(ENGINE), parent_config=str(BASE / 'config.json'),
                    source_scene=str(scene), dynamic_ply_files=config['followup_scene_spec']['actor_ply_paths'],
                    canonical_identity_fixed_before_training=True, masks=frame_rows,
                    warm_start=False, source_inputs_read_only=True, formal_updates=5000, learning_rate_horizon=30000,
                    config_changes=changes, smoke_accounting='CPU preparation only. No launch or queue change. If selected later, lifecycle smoke/full-restore/pilot acceptance must be recorded separately.',
                    inputs_role='49observedtrain BUILD; no heldout', no_subagents=True,
                    shutdown_allowed=False, human_verdict=None, quality_accepted=False)
    write(ROOT / 'identity_contract.json', contract)
    receipt = dict(status='PASS_CPU_PREPARATION_IDENTITY_AND_PIXEL_UNION', GPU_used=False,
                   training_updates=0, formal_training_started=False, historical_run_found=False,
                   constructed_old_engine_subset=True, root=str(ROOT), engine=str(ENGINE),
                   dynamic_actor_aliases=[5, 37], dynamic_rows_expected=119810, dynamic_source_real_points=[21358, 98452],
                   frames=len(frame_rows), actor_PLY_byte_exact=True, actor_asset_files_byte_exact=True,
                   all49_native_and480_union_exact=True, all49_dynamic_RGB_exact=True,
                   a5_dynamic_input=True, retained_aliases=[26, 18, 6], source_files_before=before,
                   source_files_after=after, old_files_unchanged=True,
                   config_only_top_level_changes=sorted(changes), config_only_scene_spec_changes=sorted(spec_changes),
                   config_sha256=sha(ROOT / 'config.json'), identity_contract_sha256=sha(ROOT / 'identity_contract.json'),
                   formal_updates=5000, learning_rate_horizon=30000, warm_start=False,
                   automatic_shutdown_authorized=False)
    write(ROOT / 'preparation_receipt.json', receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key not in ['source_files_before', 'source_files_after']}, indent=2))




def audit_native_initialization():
    # Read source/PLY fields on CPU. Never import the engine or torch/CUDA.
    from plyfile import PlyData
    from scipy.spatial import cKDTree
    import ast
    scene = ROOT / 'inputs/scene0008_089_137'
    # ROOT.parent is campaign; use an explicit sibling for this frozen reference.
    support_scene = ROOT.parent / 'scene008_a5_support_ablation_20261106_r1/support/inputs/scene0008_089_137'
    def vertices(path):
        return PlyData.read(path)['vertex'].data
    def xyz(rows):
        return np.column_stack([rows[k] for k in ('x','y','z')]).astype(np.float64)
    def keys(rows):
        out=np.empty(len(rows),dtype=np.dtype([(k,rows.dtype.fields[k][0]) for k in ('x','y','z','nx')]))
        for k in out.dtype.names: out[k]=rows[k]
        return out
    def stats(v):
        a=np.asarray(v,dtype=np.float64)
        return dict(min=float(a.min()),median=float(np.median(a)),max=float(a.max()),mean=float(a.mean()))
    raw = vertices(scene/'points3d.ply')
    a5 = vertices(scene/'points3d_dynamic_0.ply')
    a37 = vertices(scene/'points3d_dynamic_1.ply')
    support = vertices(support_scene/'points3d.ply')
    assert len(raw)==600000 and len(a5)==21358 and len(a37)==98452 and len(support)==619916
    before={str(p):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sha256=sha(p)) for p in
            [scene/'points3d.ply',support_scene/'points3d.ply',OLD/'points3d_dynamic_0.ply',OLD/'points3d_dynamic_4.ply',ENGINE/'scene/gaussian_model.py',ENGINE/'scene/waymo_loader2.py',ENGINE/'gaussian_renderer/__init__.py']}
    assert sha(ENGINE/'scene/gaussian_model.py')=='e5da10984f8836da1ac51c975823c3ca89bbf8c56f45ec4d4dcb071c06aafa72'
    assert sha(ENGINE/'gaussian_renderer/__init__.py')=='35b2863da1bf9173812207dbd06fbbecdb88929f95b50a7968db3b5b2aa60ced'
    full_keys=keys(a5)
    assert len(np.unique(full_keys))==21358
    a5_support = support[np.isin(keys(support),full_keys)]
    assert len(a5_support)==21412 and len(np.unique(keys(a5_support)))==21358
    pooled=np.concatenate([xyz(a5),xyz(a37)])
    keep_branch=cKDTree(pooled).query(xyz(raw),k=1,eps=0,distance_upper_bound=.1,workers=1)[0]>=.1
    keep_support=cKDTree(xyz(a37)).query(xyz(support),k=1,eps=0,distance_upper_bound=.1,workers=1)[0]>=.1
    support_static=support[keep_support]
    assert len(support_static)==611588
    assert int(np.isin(keys(support_static),full_keys).sum())==21412
    a5_original=np.isin(keys(raw),full_keys)
    assert int(a5_original.sum())==1496 and not keep_branch[a5_original].any()
    transform=np.loadtxt(scene/'transform_matrix.txt')
    support_transform=np.loadtxt(support_scene/'transform_matrix.txt')
    assert np.array_equal(transform,support_transform)
    normalize=lambda points: (np.pad(points,((0,0),(0,1)),constant_values=1)@transform.T)[:,:3].astype(np.float32)
    normalized_a5=normalize(xyz(a5))
    # Map each unique a5 source observation into the support-static rows.
    unique_support,indices=np.unique(keys(a5_support),return_index=True)
    order=np.searchsorted(unique_support,full_keys)
    mapped=a5_support[indices[order]]
    assert np.array_equal(keys(mapped),full_keys)
    assert np.array_equal(normalize(xyz(mapped)),normalized_a5)
    assert np.array_equal(mapped['nx'].astype(np.float32),a5['nx'].astype(np.float32))
    config=json.loads((ROOT/'config.json').read_text())
    assert config['random_init_point']==config['dynamic_random_init_point']==0
    span=config['frame_interval']*48
    sigma_t=float(np.sqrt(span*config['t_init']))
    assert span==.96 and config['t_init']==.1
    normalized_support=normalize(xyz(support_static))
    normalized_pooled=normalize(pooled)
    # Descriptive CPU KNN-domain check only; do not claim CUDA distCUDA2 equivalence.
    d_static=cKDTree(normalized_support).query(normalized_a5,k=4,workers=1)[0][:,1:]
    d_dynamic=cKDTree(normalized_pooled).query(normalized_a5,k=4,workers=1)[0][:,1:]
    proxy_static=np.sqrt(np.maximum((d_static*d_static).mean(1),1e-7))
    proxy_dynamic=np.sqrt(np.maximum((d_dynamic*d_dynamic).mean(1),1e-7))
    model=(ENGINE/'scene/gaussian_model.py').read_text()
    parsed=ast.parse(model)
    method_lines={}
    for cls in parsed.body:
        if isinstance(cls,ast.ClassDef) and cls.name in ('GaussianModel','Dynamic_GaussianModel'):
            for f in cls.body:
                if isinstance(f,ast.FunctionDef) and f.name in ('create_from_pcd','initialize'):
                    method_lines[cls.name+'.'+f.name]={'line':f.lineno,'end_line':f.end_lineno}
    after={p:dict(bytes=Path(p).stat().st_size,mtime_ns=Path(p).stat().st_mtime_ns,sha256=sha(Path(p))) for p in before}
    assert before==after
    receipt=dict(status='PASS_CPU_KEEP_ONLY_BRANCH_INPUT_PREPARED_NOT_LAUNCHED',root=str(ROOT),GPU_used=False,training_updates=0,
       trigger='Not queued. Reconsider only after current staticRGB2x2 results; no automatic launch.',
       dynamic_actor_aliases=[5,37],keep_only=True,native_delete_removes=[5,37],a37_only_delete_solution=False,
       original_scene_rows=len(raw),native_static_rows=int(keep_branch.sum()),removed_by_original_0p1m_rule=int((~keep_branch).sum()),
       a5_original_sampled_rows_removed=1496,native_dynamic_rows=119810,dynamic_source_real_points=[21358,98452],
       fullsupport_reference_static_rows=611588,fullsupport_a5_matching_rows=21412,fullsupport_a5_unique_xyz_time=21358,
       native_dynamic_a5_unique_xyz_time=21358,duplicate_row_difference=54,
       normalized_a5_unique_xyz_time_equal=True,normalization_transform_equal=True,model_time_field='nx, original loader normals[:,0]',
       initial_rule=dict(time='source pcd.time unchanged for both branches with random point counts0',velocity=[0,0,0],
                         sigma_t=sigma_t,log_sigma_t=float(np.log(sigma_t)),rotation=[1,0,0,0],opacity=.01,
                         RGB='native loader sets colors0 for both branches',
                         spatial_scale='Both use sqrt(clamp_min(distCUDA2(points),1e-7)), stored in log scale; KNN pool differs'),
       KNN_domains=dict(static_fullsupport='611588 retained static points including a5; a37 excluded',
                        candidate_dynamic='119810 pooled full a5+a37 points; static background excluded',
                        CPU_three_neighbor_proxy_static=stats(proxy_static),CPU_three_neighbor_proxy_dynamic=stats(proxy_dynamic),
                        CPU_proxy_not_native_CUDA_initialization=True,pointwise_proxy_difference=stats(np.abs(proxy_static-proxy_dynamic))),
       source_method_lines=method_lines,engine=str(ENGINE),engine_unmodified=True,source_files_before=before,source_files_after=after,
       scientific_boundary='Branch-responsibility/targets/representation bundle, not a single-scalar causal isolation; 54duplicate rows and KNN domain differ.',
       pilot_first=dict(raw_ids=[97,105,109],checkpoint_nodes=[1000,5000],native_automatic_full49_disabled=True,
                        full49_only_after_pilot_acceptance=True,training_frame_sampling_unchanged='raw89..137',
                        future_wrapper_must_snapshot_restore_RNG_around_independent_evaluation=True,
                        CPU_does_not_prove_future_runtime_sampler_or_RNG_preservation=True),
       lifecycle='Before any future long run, verify real save/evaluation/full-state restore/next-stage entry; no GPU validation performed here',
       weights_copied=False,quality_accepted=False,human_verdict=None,shutdown=False,use_reset_card=False,
       preparation_script_sha256=sha(Path(__file__)))
    write(ROOT/'branch_control_cpu_audit.json',receipt)
    contract=dict(task_id='SCENE008-A5-BRANCH-CONTROL-20261107-R1',status='PREPARED_CPU_ONLY_NOT_QUEUED',
                  conditional_on='Current staticRGB supervision2x2 result reviewed; no automatic launch',formal_stop=5000,
                  learning_rate_horizon=30000,dynamic_actor_aliases=[5,37],KEEP_only=True,
                  delete='Native render_s deletes botha5+a37; never a5-preservation metric',
                  reference='scene008_a5_support_ablation_20261106_r1/support fresh5K, a5 static',
                  mathematical_change='Native identity/branch assignment bundle with native union RGB/opacity target and branch-specific priors; not an isolated single weight',
                  input_audit=str(ROOT/'branch_control_cpu_audit.json'),input_audit_sha256=sha(ROOT/'branch_control_cpu_audit.json'),
                  evaluation_plan=receipt['pilot_first'],native_engine_unmodified=True,training_started=False,
                  no_launch_controller_created=True,no_large_checkpoint_copied=True,quality_accepted=False,human_verdict=None)
    write(ROOT/'conditional_fallback_contract.json',contract)
    print(json.dumps({k:receipt[k] for k in ['status','native_static_rows','removed_by_original_0p1m_rule','native_dynamic_rows','duplicate_row_difference','initial_rule']},indent=2))


if __name__ == '__main__':
    main()
    audit_native_initialization()
