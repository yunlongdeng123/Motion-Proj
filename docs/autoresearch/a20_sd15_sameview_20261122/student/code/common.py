"""CPU-safe contracts, independent of torch/native rendering imports."""
from pathlib import Path
import hashlib
import json
import math
import os
import numpy as np

TRACK = 'bzFbWRrSGOgEaXxnuGV0tw'
BUILD = (86, 110)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def ref(path):
    p = Path(path).resolve()
    return dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + '.tmp')
    q.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    q.replace(p)


def bank(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def validate_contract(c):
    if c['build_raws'] != [86, 110] or c['heldout_raws'] != [134,179,198]:
        raise ValueError('Frozen BUILD/heldout contract changed')
    if c['views']['synthetic_resolution'] != [512,512] or c['views']['build_resolution'] != [1152,768]:
        raise ValueError('Native-resolution supervision is required')
    if c['stages']['geometry']['formal_steps'] != 40 or c['stages']['material']['formal_steps'] != 80:
        raise ValueError('The authorized formal budget is exactly 40+80')
    if c['stages']['smoke']['steps'] != 20 or c['student']['density_control'] or c['student']['pruning']:
        raise ValueError('Unapproved smoke/density policy')
    if c['joint']['optimization'] or c['loss']['real_U_is_negative'] or c['teacher']['downsampling']:
        raise ValueError('Unapproved supervision domain')
    if c['student']['missing_normal_jitter_m'] != 0:
        raise ValueError('Formal independent sampling must not contain artificial perturbation')
    for key,expected in [('teacher_depth_alpha_minimum',.90),('teacher_material_alpha_minimum',.95),('student_material_alpha_minimum',.95)]:
        if c['loss'][key]!=expected:raise ValueError('Frozen confidence mask changed: '+key)
    for key in ('generated_missing_masked_edge_L1','generated_missing_HF_residual_L1','real_V_masked_edge_L1','real_V_HF_residual_L1'):
        if c['loss']['material'][key]!=.1:raise ValueError('Frozen detail-loss weight changed: '+key)
    if c['loss']['material_fixed_domain_required_coverage_fraction'] != .99:
        raise ValueError('Material coverage must use the predeclared fixed-domain 99% gate')
    return c


def validate_packet(a, allow_opacity_boundary=False):
    n = len(a['xyz'])
    for key, shape in dict(xyz=(n,3), scales=(n,3), rotation=(n,4), coefficients=(n,16,3),
                           opacity=(n,), observed_protected=(n,), missing=(n,), is_opaque_anchor=(n,)).items():
        if a[key].shape != shape or not np.isfinite(a[key]).all():
            raise ValueError('Invalid Gaussian field '+key)
    invalid_opacity=((a['opacity'] < 0)|(a['opacity'] > 1)) if allow_opacity_boundary else ((a['opacity'] <= 0)|(a['opacity'] >= 1))
    if (a['scales'] <= 0).any() or invalid_opacity.any():
        raise ValueError('Invalid activated Gaussian parameters')
    if np.max(np.abs(np.linalg.norm(a['rotation'],axis=1)-1)) > 1e-4:
        raise ValueError('Non-unit wxyz quaternion')
    if np.any(a['missing'] == a['observed_protected']):
        raise ValueError('Missing and observed must form disjoint complete support')
    if not np.all(a['observed_protected'][a['is_opaque_anchor'].astype(bool)]):
        raise ValueError('Opaque anchors must be protected')
    return n


def synthetic_specs(c):
    v = c['views']; result = []
    for i, az in enumerate(v['synthetic_azimuth_degrees']):
        a = math.radians(az); e = math.radians(v['synthetic_elevation_degrees'])
        target = np.asarray(v['synthetic_target_actor_m'],float)
        eye = target + v['synthetic_distance_m']*np.array([math.cos(e)*math.cos(a),math.cos(e)*math.sin(a),math.sin(e)])
        forward = target-eye; forward /= np.linalg.norm(forward)
        right = np.cross(forward,[0.,0.,1.]); right /= np.linalg.norm(right)
        down = np.cross(forward,right)
        W = np.eye(4); W[:3,:3] = np.stack([right,down,forward]); W[:3,3] = -W[:3,:3]@eye
        near, far = .1, 100.
        P = np.zeros((4,4)); f = 1/math.tan(math.radians(v['synthetic_fov_degrees'])/2)
        P[0,0]=P[1,1]=f; P[2,2]=far/(far-near); P[2,3]=-far*near/(far-near); P[3,2]=1
        result.append(dict(key=f'novel_{i:02d}', azimuth_degrees=az, image_width=512,image_height=512,
            fov_degrees=v['synthetic_fov_degrees'],camera_center_actor_m=eye.tolist(),
            camera_from_actor=W.tolist(),projection_matrix=P.tolist()))
    return result


def check_approval(path, contract, stage, bindings):
    a = read(path)
    if a.get('schema') != 'a20.object_centric_distillation.approval.v1' or a.get('approved') is not True:
        raise ValueError('Root approval is absent')
    if a.get('contract_sha256') != sha(contract) or stage not in a.get('allowed_phases',[]):
        raise ValueError('Root approval does not bind this contract/phase')
    for key, digest in bindings.items():
        if a.get(key) != digest:
            raise ValueError('Approval binding mismatch: '+key)
    if stage == 'material' and a.get('geometry_gate_passed') is not True:
        raise ValueError('The explicit geometry grey gate has not passed')
    return a


def cpu_environment():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[k]='2'
