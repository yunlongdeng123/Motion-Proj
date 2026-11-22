"""CPU-safe contracts for a single, source-bound, editable a20 Gaussian field."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np

BUILD = (86, 110)
SEALED = (134, 179, 198)
FIELDS = ('xyz', 'scales', 'rotation', 'coefficients', 'opacity')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def bank(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


DEFAULT_CONFIG = dict(
    schema='a20.unified_field.training_contract.v1', build_raws=list(BUILD), sealed_raws=list(SEALED),
    seed=101, stages=dict(smoke=10, geometry=40, appearance=80, joint=40),
    learning_rates=dict(xyz=0.0005, log_scales=0.001, raw_rotation=0.0005, opacity=0.005, dc=0.005),
    loss=dict(real_rgb=1., retained_K_rgb=1., retained_K_alpha=1., silhouette_positive=0.2, silhouette_negative=0.2, teacher_background=0.2,
              lidar_range=2., lidar_alpha=0.5, teacher_alpha=0.5, teacher_depth=0.2,
              teacher_rgb=0.25, parameter_geometry=0.02, parameter_appearance=0.02,
              delta_smoothness=0.002, duplicate_opacity=0.002),
    lidar=dict(opaque_group=2, glass_group=1, stable_alpha=0.95, training_denominator_floor=0.05, huber_delta_m=0.03,
               expected_opaque_count=246, expected_all_count=1030),
    teacher=dict(stable_alpha=0.95, training_denominator_floor=0.05, unknown_attribution_minimum=0.95,
                 background_support_alpha=0.01, background_dilation_pixels=8, background_topk_pixels=1024,
                 real_support_margin_m=0.075),
    retained_K=dict(topk_pixels=1024, changed_threshold=1/255),
    regularizer=dict(xyz_scale_m=0.05, log_scale_scale=0.2, quaternion_scale=0.1,
                     opacity_scale=0.2, dc_scale=0.5, confidence_floor=0.1,
                     safety_xyz_bound_m=0.3, safety_scale_ratio_bounds=[0.5, 2.]),
    redundancy=dict(refresh_every=10, cross_source_radius_m=0.05, conflict_range_m=0.075,
                    inactive_opacity_max=0.01, support_contribution_max=0.001,
                    conflict_fraction_minimum=0.8, novel_outside_view_minimum=2,
                    novel_outside_raw_contribution_minimum=0.001, automatic_global_retirement=True),
    geometry_gate=dict(visible_alpha95_fraction_minimum=0.99, opaque_stable_count_minimum=246,
                       opaque_mae_m_maximum=0.075, opaque_p90_m_maximum=0.15,
                       novel_unknown_alpha95_fraction_minimum=0.99),
    raster=dict(real_resolution=[1152, 768], synthetic_resolution=[512, 512],
                shared_scene_depth_sort=True, per_camera_row_selection=False, rgb_alpha_division=False),
    supervision=dict(real_V_rgb='isolated actual native actor raster', real_V_positive_alpha='isolated actual native actor raster',
                     retained_K='shared native scene versus frozen context-only DELETE',
                     teacher_background='outside frozen pure-T alpha>.01 dilated8px; alpha squared only'),
    freeze=['road', 'camera', 'trajectory', 'native_static', 'native_dynamic', 'other_actors', 'environment', 'SH_rest'],
)


class Sources:
    def __init__(self, path_maps=()):
        self.maps = sorted([tuple(v.split('=', 1)) for v in path_maps], key=lambda p: -len(p[0]))
        self.opened = []

    def path(self, value):
        value = str(value).replace('\\', '/')
        for old, new in self.maps:
            old = old.replace('\\', '/').rstrip('/')
            if value.casefold() == old.casefold() or value.casefold().startswith(old.casefold() + '/'):
                value = new.rstrip('/') + value[len(old):]; break
        return Path(value)

    def get(self, item, role, required=True):
        if isinstance(item, (str, Path)):
            item = dict(path=str(item))
        if not item or not item.get('path'):
            if required: raise ValueError('Missing source: ' + role)
            return None
        value = item.get('runtime_path', item['path']) if os.name != 'nt' else item['path']
        path = self.path(value)
        if not path.is_file():
            if required: raise FileNotFoundError(role + ': ' + str(path))
            return None
        digest = sha(path)
        if item.get('sha256') and digest != item['sha256']:
            raise ValueError('Source hash mismatch: ' + role)
        self.opened.append(dict(path=str(path.resolve()), sha256=digest, purpose=role, raw=None))
        return path


def validate_actor(a):
    n = len(a['xyz'])
    shapes = dict(xyz=(n, 3), scales=(n, 3), rotation=(n, 4), coefficients=(n, 16, 3), opacity=(n,))
    for key, shape in shapes.items():
        if key not in a or a[key].shape != shape or a[key].dtype != np.float32 or not np.isfinite(a[key]).all():
            raise ValueError('Invalid float32 activated Gaussian field: ' + key)
    if (a['scales'] <= 0).any() or ((a['opacity'] < 0) | (a['opacity'] > 1)).any():
        raise ValueError('Invalid activated scale/opacity')
    if np.max(np.abs(np.linalg.norm(a['rotation'], axis=1) - 1)) > 1e-4:
        raise ValueError('Expected unit wxyz quaternions')
    for key in ('source_id', 'source_row_id', 'native_global_row_id'):
        if key not in a or a[key].shape != (n,) or a[key].dtype.kind not in 'iu':
            raise ValueError('Missing strict row identity: ' + key)
    if not np.isin(a['source_id'], [0, 1]).all():
        raise ValueError('source_id must be old=0 or TRELLIS=1')
    for sid in (0, 1):
        ids = a['source_row_id'][a['source_id'] == sid]
        if len(ids) != len(np.unique(ids)) or (ids < 0).any():
            raise ValueError('Source row identity duplicated or negative')
    tids = a['source_row_id'][a['source_id'] == 1]
    if not np.array_equal(np.sort(tids), np.arange(410080)):
        raise ValueError('All 410080 native TRELLIS rows must initialize the field exactly once')
    for key in ('geometry_confidence', 'appearance_confidence', 'teacher_unknown', 'teacher_prior_weight'):
        if key not in a or a[key].shape != (n,) or not np.isfinite(a[key]).all():
            raise ValueError('Missing confidence/teacher metadata: ' + key)
    for key in ('geometry_confidence', 'appearance_confidence', 'teacher_prior_weight'):
        if ((a[key] < 0) | (a[key] > 1)).any(): raise ValueError('Confidence outside [0,1]: ' + key)
    if np.any(a['teacher_unknown'][a['source_id'] == 0]):
        raise ValueError('Frozen teacher unknown attribution must reference source TRELLIS only')
    return n


def load_inputs(args):
    maps = list(args.path_map)
    if args.path_map_file:
        maps += [x['from'] + '=' + x['to'] for x in read(args.path_map_file)['path_map']]
    s = Sources(maps)
    init_path = s.get(args.init, 'unified source-bound initialization')
    init = bank(init_path); n = validate_actor(init)
    manifest_path = s.get(args.manifest, 'unified initialization manifest')
    manifest = read(manifest_path)
    source_manifest_path = s.get(args.source_manifest, 'native source bindings and fixed BUILD domains')
    native = read(source_manifest_path)
    fixed = native['fixed_domains']
    teacher_item = args.teacher or native.get('source_T_npz')
    if teacher_item is None:
        pm = read(s.get(native['prior_manifest'], 'frozen TRELLIS source manifest'))
        teacher_item = pm['source']
    teacher_path = s.get(teacher_item, 'frozen full native TRELLIS teacher')
    teacher = bank(teacher_path)
    for key in FIELDS:
        if teacher[key].shape != init[key][init['source_id'] == 1].shape:
            raise ValueError('Teacher shape does not match full native TRELLIS source: ' + key)
        take = np.argsort(init['source_row_id'][init['source_id'] == 1])
        if not np.array_equal(init[key][init['source_id'] == 1][take], teacher[key]):
            raise ValueError('TRELLIS initialization must exactly preserve activated source fields: ' + key)
    config = read(args.config) if args.config else json.loads(json.dumps(DEFAULT_CONFIG))
    if config['build_raws'] != list(BUILD) or config['sealed_raws'] != list(SEALED):
        raise ValueError('Only BUILD86/110 may be opened')
    for raw in BUILD:
        f = fixed['real_V'][f'{raw:03d}']
        for key in ('mapping', 'visible_mask', 'unknown_potential', 'known_background'):
            s.get(f[key], f'fixed {raw} {key}')
        s.get(fixed['real_K_labels'][f'{raw:03d}'], f'fixed {raw} categorical K labels')
    measured_path = s.get(fixed['measured_returns']['grouped_returns_npz'], 'all1030 frozen measured returns')
    measured = bank(measured_path)
    if len(measured['xyz_canonical_m']) != 1030 or set(measured['source_raw'].tolist()) != set(BUILD):
        raise ValueError('Expected all1030 fixed BUILD measured returns')
    if int((measured['RGB_region_group'] == 2).sum()) != 246:
        raise ValueError('Expected exactly 246 opaque returns; glass cannot become exterior shell')
    return dict(sources=s, init=init, init_path=init_path, manifest=manifest, manifest_path=manifest_path,
                native=native, native_path=source_manifest_path, fixed=fixed, teacher=teacher,
                teacher_path=teacher_path, measured=measured, measured_path=measured_path, config=config, rows=n)


def preflight(args, inp):
    a = inp['init']
    result = dict(schema='a20.unified_field.cpu_preflight.v1', passed=True, GPU_used=False,
                  unified_rows=inp['rows'], old_rows=int((a['source_id'] == 0).sum()), trellis_rows=410080,
                  teacher_unknown_rows=int(a['teacher_unknown'].sum()),
                  initialized_fields=list(FIELDS), editable_geometry=['xyz', 'scales', 'rotation', 'opacity'],
                  editable_appearance=['SH_DC'], source_bound=True, full_native_T=True,
                  source_initialization=ref(inp['init_path']), initialization_manifest=ref(inp['manifest_path']),
                  source_manifest=ref(inp['native_path']), frozen_teacher=ref(inp['teacher_path']),
                  expected_opaque_returns=246, glass_used_as_shell=False, build_raws=list(BUILD),
                  heldout_frames_opened=[], previous_P0_or_monotone_T_filter_used=False,
                  per_camera_actor_selection=False, config=inp['config'], read_ledger=inp['sources'].opened)
    write(args.output / 'cpu_preflight.json', result)
    write(args.output / 'training_contract.json', inp['config'])
    return result
