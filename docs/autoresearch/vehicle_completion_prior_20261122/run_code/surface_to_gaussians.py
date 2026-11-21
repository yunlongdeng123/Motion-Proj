"""CPU：将生成纹理表面转换为米制 actor Gaussians，保留逐点表面来源。"""
from pathlib import Path
import argparse
import json
import os
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '2')

import numpy as np
from PIL import Image
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

SAMPLE_COUNT = 25_000
SEED = 101
C0 = 0.28209479177387814
GENERATED_SURFACE_BRANCH = 4


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def finite_vector(value, length, name):
    array = np.asarray(value, np.float64)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(name+' must be a finite vector')
    return array


def coordinate_contract(prepared, texture_receipt):
    """明确合并两层可逆归一化；不再次撤销官方内部 rotation。"""
    center_actor = finite_vector(prepared['actor_center_m'], 3, 'actor_center_m')
    metres = float(prepared['metres_per_texture_unit'])
    permutation = np.asarray(prepared['axis_permutation'], np.int64)
    if permutation.shape != (3,) or sorted(permutation.tolist()) != [0, 1, 2]:
        raise ValueError('axis_permutation must be a permutation of 0,1,2')
    if permutation.tolist() != [0, 2, 1]:
        raise ValueError('this run requires axis_permutation=[0,2,1]')
    official = texture_receipt['official_mesh_normalization']
    center_official = finite_vector(official['center_in_input_coordinates'], 3, 'official center')
    scale_official = float(official['scale_input_to_output'])
    if not (np.isfinite(metres) and metres > 0 and np.isfinite(scale_official) and scale_official > 0):
        raise ValueError('both normalization scales must be finite and positive')
    if 'undone' not in str(official.get('internal_axis_rotation', '')).lower():
        raise ValueError('receipt must confirm internal rotation was undone by postprocess_mesh')
    permutation_matrix = np.eye(3)[permutation]
    reflected = bool(np.linalg.det(permutation_matrix) < 0)
    if bool(prepared.get('reflection_winding_reversed')) != reflected:
        raise ValueError('prepared winding record does not match axis permutation parity')
    actor_from_output_before_post = np.eye(4)
    actor_from_output_before_post[:3, :3] = permutation_matrix * (metres / scale_official)
    actor_from_output_before_post[:3, 3] = center_actor + metres*center_official[permutation]
    post_value = prepared.get('post_actor_transform')
    has_post = post_value is not None
    post = np.eye(4) if not has_post else np.asarray(post_value, np.float64)
    if (post.shape != (4,4) or not np.isfinite(post).all()
            or not np.allclose(post[3], [0,0,0,1], atol=1e-10, rtol=0)
            or not np.allclose(post[:3,:3].T@post[:3,:3], np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(post[:3,:3]), 1, atol=1e-8, rtol=0)):
        raise ValueError('post_actor_transform must be a finite proper rigid 4x4 transform')
    # 仅派生绑定左乘该刚体修正；无字段时沿用原矩阵，UV 与原始 GLB 均不改写。
    actor_from_output = post@actor_from_output_before_post if has_post else actor_from_output_before_post
    heading_fix = dict(applied=has_post,post_actor_transform=post.tolist(),
        description=prepared.get('axis_heading_fix', 'semantic front +X heading correction' if has_post else 'none'),
        operation='left-multiply derived actor binding; positions and normals follow the same proper rigid transform',
        geometry_optimization_updates=0,raw_mesh_or_glb_modified=False,texture_uv_modified=False)
    return dict(actor_from_output=actor_from_output, output_from_actor=np.linalg.inv(actor_from_output),
                actor_from_output_before_post=actor_from_output_before_post,post_actor_transform=post,
                axis_heading_fix=heading_fix,
                center_actor=center_actor, metres=metres, permutation=permutation,
                center_official=center_official, scale_official=scale_official,
                reflected=reflected)


def apply_transform(xyz, matrix):
    return np.asarray(xyz, np.float64) @ matrix[:3, :3].T + matrix[:3, 3]


def output_to_actor(xyz, contract):
    return apply_transform(xyz, contract['actor_from_output'])


def actor_to_output(xyz, contract):
    return apply_transform(xyz, contract['output_from_actor'])


def load_mesh(path):
    with np.load(path, allow_pickle=False) as data:
        vertices = np.asarray(data['vertices'], np.float64)
        raw_faces = np.asarray(data['faces'])
        uv = np.asarray(data['uv'], np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError('vertices must be finite Vx3')
    if raw_faces.ndim != 2 or raw_faces.shape[1] != 3 or not len(raw_faces):
        raise ValueError('faces must be nonempty Fx3 triangles')
    if not np.issubdtype(raw_faces.dtype, np.integer):
        raise ValueError('faces must have integer vertex indices')
    faces = raw_faces.astype(np.int64, copy=False)
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('mesh vertex index out of bounds')
    if uv.shape != (len(vertices), 2) and uv.shape != (len(faces), 3, 2):
        raise ValueError('UV must be Vx2 or Fx3x2; official output splits seam vertices')
    if not np.isfinite(uv).all() or (uv < -1e-6).any() or (uv > 1+1e-6).any():
        raise ValueError('this atlas requires finite UV in [0,1]')
    return vertices, faces, np.clip(uv, 0, 1)


def bilinear_rgba(texture, uv):
    """UV 原点左下；PNG 原点左上；不预乘 alpha，不改 sRGB 编码。"""
    image = np.asarray(texture)
    if image.ndim != 3 or image.shape[2] != 4 or image.dtype != np.uint8:
        raise ValueError('texture must be uint8 HxWx4 RGBA')
    if not np.isfinite(uv).all() or (uv < 0).any() or (uv > 1).any():
        raise ValueError('UV outside clamped atlas')
    height, width = image.shape[:2]
    if height < 1 or width < 1:
        raise ValueError('empty texture')
    x = uv[:, 0]*(width-1)
    y = (1-uv[:, 1])*(height-1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0+1, width-1), np.minimum(y0+1, height-1)
    tx, ty = (x-x0)[:, None], (y-y0)[:, None]
    top = image[y0, x0].astype(np.float64)*(1-tx) + image[y0, x1]*tx
    bottom = image[y1, x0].astype(np.float64)*(1-tx) + image[y1, x1]*tx
    return (top*(1-ty)+bottom*ty)/255


def normal_quaternions(normals):
    """列 2 为法向；四元数 wxyz，与上轮 Vehicle/3DGS 一致。"""
    normals = np.asarray(normals, np.float64)
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError('normals must be Nx3')
    lengths = np.linalg.norm(normals, axis=1)
    if not np.isfinite(normals).all() or (lengths <= 0).any():
        raise ValueError('invalid normal')
    normals = normals/lengths[:, None]
    reference = np.eye(3)[np.argmin(np.abs(normals), axis=1)]
    tangent0 = np.cross(reference, normals)
    tangent0 /= np.linalg.norm(tangent0, axis=1)[:, None]
    tangent1 = np.cross(normals, tangent0)
    matrices = np.stack([tangent0, tangent1, normals], axis=-1)
    xyzw = Rotation.from_matrix(matrices).as_quat()
    wxyz = xyzw[:, [3, 0, 1, 2]]
    wxyz[wxyz[:, 0] < 0] *= -1
    return wxyz


def farthest_subset(points, count):
    """单次过采样池的确定性最远子集；选既有表面点，不添加或复制点。"""
    if len(points) < count:
        raise ValueError('not enough candidates for exact farthest subset')
    if len(points) == count:
        return np.arange(count, dtype=np.int64)
    center = points.mean(axis=0)
    chosen = int(np.argmax(np.einsum('ij,ij->i', points-center, points-center)))
    distance = np.full(len(points), np.inf)
    selected = np.empty(count, np.int64)
    for i in range(count):
        selected[i] = chosen
        delta = points-points[chosen]
        np.minimum(distance, np.einsum('ij,ij->i', delta, delta), out=distance)
        distance[selected[:i+1]] = -1
        chosen = int(np.argmax(distance))
    return selected


def even_area_samples(vertices_actor, faces, valid_faces, count, seed, total_area):
    """官方近均匀面积采样；不足时只做一次同 seed 过采样，不重复填充。"""
    face_candidates = np.flatnonzero(valid_faces)
    mesh = trimesh.Trimesh(vertices=vertices_actor, faces=faces[valid_faces], process=False, validate=False)
    points, local_face = trimesh.sample.sample_surface_even(mesh, count=count, seed=seed)
    initial_count = len(points)
    fallback_count = None
    fallback_requested = None
    farthest_used = False
    pool = 'initial'
    if initial_count < count:
        # 单次工程补采样，保留相同 seed；不做参数/候选搜索。
        fallback_requested = 2*count
        extra_points, extra_face = trimesh.sample.sample_surface_even(mesh, count=fallback_requested, seed=seed)
        fallback_count = len(extra_points)
        if fallback_count > initial_count:
            points, local_face = extra_points, extra_face
            pool = 'single_oversampling_fallback'
        if len(points) > count:
            ids = farthest_subset(points, count)
            points, local_face = points[ids], local_face[ids]
            farthest_used = True
    if not len(points):
        raise ValueError('near-uniform sampler returned no points')
    actual_count = len(points)
    nearest = cKDTree(points).query(points, k=2, workers=1)[0][:, 1] if actual_count > 1 else np.zeros(1)
    quantiles = [0, .01, .05, .25, .5, .75, .95, .99, 1]
    names = ['p00','p01','p05','p25','p50','p75','p95','p99','p100']
    h = np.sqrt(total_area/actual_count)
    metadata = dict(
        method='trimesh.sample.sample_surface_even',trimesh_version=trimesh.__version__,
        requested_count=count,actual_count=actual_count,exact_requested_count=(actual_count==count),
        seed=seed,initial_requested=count,initial_returned=initial_count,
        initial_rejection_radius_m=float(np.sqrt(total_area/(3*count))),
        single_oversampling_requested=fallback_requested,single_oversampling_returned=fallback_count,
        selected_pool=pool,deterministic_farthest_subset_used=farthest_used,
        duplicated_points_added=False,
        nearest_neighbor_distance_m={name:float(value) for name,value in zip(names,np.quantile(nearest,quantiles))},
        nearest_neighbor_distance_over_h={name:float(value) for name,value in zip(names,np.quantile(nearest/h,quantiles))},
        renderer_alpha_coverage_checked=False,
        coverage_verdict='not evaluated by CPU uniformity; requires actual renderer alpha check',
    )
    return np.asarray(points,np.float64),face_candidates[np.asarray(local_face,np.int64)],nearest,metadata


def sample_surface(vertices_output, faces, uv, texture, contract, *, count=SAMPLE_COUNT, seed=SEED):
    """米制近均匀表面采样；face_id/barycentric 对应原始 generated_mesh.faces。"""
    if not isinstance(count, int) or count <= 0:
        raise ValueError('sample count must be positive integer')
    vertices_actor = output_to_actor(vertices_output, contract)
    triangle = vertices_actor[faces]
    cross = np.cross(triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0])
    cross_length = np.linalg.norm(cross, axis=1)
    area = cross_length*.5
    if not np.isfinite(area).all() or not area.sum() > 0:
        raise ValueError('mesh has no finite positive surface area')
    valid = area > 0
    total_area = float(area.sum())
    xyz_actor,face_id,nearest,sampling = even_area_samples(vertices_actor,faces,valid,count,seed,total_area)
    count = len(xyz_actor)
    barycentric = trimesh.triangles.points_to_barycentric(triangle[face_id],xyz_actor,method='cross')
    if not np.isfinite(barycentric).all() or (barycentric < -1e-8).any() or (barycentric > 1+1e-8).any():
        raise ValueError('sampled point does not reconstruct on its source face')
    barycentric = np.clip(barycentric,0,1)
    barycentric /= barycentric.sum(axis=1,keepdims=True)
    xyz_actor = np.einsum('ni,nij->nj', barycentric, triangle[face_id])
    xyz_output = np.einsum('ni,nij->nj', barycentric, vertices_output[faces[face_id]])
    texture_input_xyz = xyz_output/contract['scale_official'] + contract['center_official']
    face_uv = uv[faces] if uv.ndim == 2 else uv
    sample_uv = np.clip(np.einsum('ni,nij->nj', barycentric, face_uv[face_id]), 0, 1)
    rgba = bilinear_rgba(texture, sample_uv)
    # 从 TRELLIS Y-up 恢复到 actor Z-up 是反射。恢复等价绕序后才定义外法向。
    normal_sign = -1.0 if contract['reflected'] else 1.0
    normals = cross[face_id] / cross_length[face_id, None] * normal_sign
    quaternions = normal_quaternions(normals)
    spacing = float(np.sqrt(total_area/count))
    scales = np.tile([.6*spacing, .6*spacing, .1*spacing], (count, 1))
    coefficients = np.zeros((count, 16, 3), np.float32)
    coefficients[:, 0] = (rgba[:, :3]-.5)/C0
    result = dict(
        xyz=xyz_actor.astype(np.float32), scales=scales.astype(np.float32),
        rotation=quaternions.astype(np.float32), coefficients=coefficients,
        opacity=np.full(count, .9, np.float32),
        source_row=np.full(count, -1, np.int64), source_branch=np.full(count, GENERATED_SURFACE_BRANCH, np.int8),
        source_raw=np.full(count, -1, np.int16),
        surface_face_id=face_id.astype(np.int64), surface_barycentric=barycentric.astype(np.float32),
        surface_normal_actor=normals.astype(np.float32), surface_uv=sample_uv.astype(np.float32),
        surface_rgb_generated_srgb=rgba[:, :3].astype(np.float32), surface_basecolor_alpha=rgba[:, 3].astype(np.float32),
        surface_point_trellis_output=xyz_output.astype(np.float32),
        surface_point_texture_input=texture_input_xyz.astype(np.float32),
        is_observed=np.zeros(count, bool), is_generated_prior=np.ones(count, bool),
        visibility_evaluated=np.asarray(False),
        surface_total_area_m2=np.float64(total_area), sample_spacing_h_m=np.float64(spacing),
        surface_nearest_neighbor_distance_m=nearest.astype(np.float32),
        surface_sampling_method=np.asarray('trimesh.sample.sample_surface_even'),
        actor_from_trellis_output_4x4=contract['actor_from_output'],
        actor_from_trellis_output_before_post_4x4=contract['actor_from_output_before_post'],
        post_actor_transform_4x4=contract['post_actor_transform'],
        axis_heading_fix_json=np.asarray(json.dumps(contract['axis_heading_fix'],ensure_ascii=False,sort_keys=True)),
        trellis_output_from_actor_4x4=contract['output_from_actor'],
        sample_seed=np.int64(seed), source_kind=np.asarray('generated_textured_surface_prior'),
        quaternion_convention=np.asarray('wxyz; local third/thin axis follows actor surface normal'),
        barycentric_convention=np.asarray('weights of original generated_mesh.faces[face_id] vertex order'),
    )
    roundtrip = actor_to_output(vertices_actor, contract)
    quaternion_matrices = Rotation.from_quat(result['rotation'][:, [1, 2, 3, 0]]).as_matrix()
    thin_axis_error = float(np.max(np.linalg.norm(quaternion_matrices[:, :, 2]-normals, axis=1)))
    metadata = dict(
        sample_count=count, requested_sample_count=sampling['requested_count'], seed=seed, surface_area_m2=total_area,
        sampling=sampling,
        axis_heading_fix=contract['axis_heading_fix'],
        spacing_h_m=spacing, tangent_sigma_m=.6*spacing, normal_sigma_m=.1*spacing,
        opacity=.9, opacity_from_texture_alpha=False,
        input_vertices=len(vertices_output), input_faces=len(faces), zero_area_faces_skipped=int((~valid).sum()),
        actor_bounds_m=[vertices_actor.min(0).tolist(), vertices_actor.max(0).tolist()],
        sampled_bounds_m=[xyz_actor.min(0).tolist(), xyz_actor.max(0).tolist()],
        mesh_normalization_roundtrip_max_abs=float(np.max(np.abs(roundtrip-vertices_output))),
        quaternion_thin_axis_max_l2=thin_axis_error,
        corrected_actor_winding_for_axis_reflection=contract['reflected'],
        uv_origin='bottom-left; image top-left; v flipped once',
        texture_interpolation='bilinear, clamped atlas; x=u*(W-1), y=(1-v)*(H-1); straight RGBA',
        texture_size=[int(texture.shape[1]), int(texture.shape[0])],
        color='basecolor sRGB stored as generated appearance prior; SH DC=(sRGB-.5)/C0, degree1..3 zero',
        pbr='PBR maps remain independent provenance; roughness/metallic are not SH channels',
        source_branch_mapping={'4':'generated textured surface prior; not an observed/native scene Gaussian'},
        observed_support_evaluated=False,
    )
    if metadata['mesh_normalization_roundtrip_max_abs'] > 1e-9 or thin_axis_error > 1e-5:
        raise ValueError('coordinate or quaternion numeric verification failed')
    return result, metadata


def pbr_sources(basecolor, receipt_path, receipt, extra_maps):
    maps = {'basecolor_rgba': str(Path(basecolor).resolve())}
    folder = Path(receipt_path).resolve().parent
    for name in receipt.get('output_files', []):
        if isinstance(name, str) and Path(name).suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.tif', '.tiff'}:
            path = (folder/name).resolve()
            if not path.is_file():
                raise FileNotFoundError('receipt PBR texture missing: '+str(path))
            maps[path.stem] = str(path)
    for item in extra_maps:
        if '=' not in item:
            raise ValueError('--pbr-map requires role=/absolute/path')
        role, path = item.split('=', 1)
        if not role or not Path(path).is_file():
            raise ValueError('invalid additional PBR map: '+item)
        maps[role] = str(Path(path).resolve())
    return dict(paths=maps, basecolor_encoding='sRGB straight RGBA',
                metallic_roughness_channels='if present: G=roughness, B=metallic; no conversion into SH',
                material_alpha_used_for_gaussian_opacity=False)


def convert(mesh_path, basecolor_path, transform_path, receipt_path, output, extra_maps=()):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('preserve existing surface conversion: '+str(output))
    prepared, receipt = read_json(transform_path), read_json(receipt_path)
    if receipt.get('status') != 'completed':
        raise ValueError('TRELLIS output receipt must be completed')
    contract = coordinate_contract(prepared, receipt)
    vertices, faces, uv = load_mesh(mesh_path)
    texture = np.asarray(Image.open(basecolor_path).convert('RGBA'))
    arrays, stats = sample_surface(vertices, faces, uv, texture, contract)
    pbr = pbr_sources(basecolor_path, receipt_path, receipt, extra_maps)
    provenance = dict(
        schema='vehicle.generated_surface_gaussians.v1',
        source_mesh=str(Path(mesh_path).resolve()), source_basecolor=str(Path(basecolor_path).resolve()),
        prepared_transform=str(Path(transform_path).resolve()), texture_run_receipt=str(Path(receipt_path).resolve()),
        sampling_protocol=dict(requested_count=SAMPLE_COUNT,actual_count=stats['sample_count'],seed=SEED,
            distribution='approximate equal area with proximity rejection; sample_surface_even',
            exact_requested_count=stats['sampling']['exact_requested_count']),
        coordinate_inverse='texture_input=trellis_output/official_scale+official_center; actor_registered=texture_input[:,[0,2,1]]*metres_per_texture_unit+actor_center_m; actor_final=post_actor_transform@actor_registered (homogeneous column vectors)',
        official_internal_rotation='already undone by postprocess_mesh; not undone again',
        axis_heading_fix=contract['axis_heading_fix'],
        actor_axes='x forward, y left, z up; metres',
        units=dict(xyz='metres', scales='positive linear Gaussian standard deviations in metres', rotation='unit quaternion wxyz', opacity='linear probability'),
        source_role='learned generated geometry and material prior; sampled rows are not real observed returns',
        source_raw_policy='all source_raw=-1 and source_row=-1; mesh face_id/barycentric provide generative provenance',
        real_visible_support_policy='not evaluated here; downstream train86/110 mask/render contribution evaluation is separate',
        pbr=pbr, source_branch=GENERATED_SURFACE_BRANCH, numpy_version=np.__version__,
        gpu_calls=0, training_updates=0, output='vehicle.npz', statistics=stats)
    arrays['pbr_sources_json'] = np.asarray(json.dumps(pbr, ensure_ascii=False, sort_keys=True))
    arrays['surface_provenance_json'] = np.asarray(json.dumps(provenance, ensure_ascii=False, sort_keys=True))
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'vehicle.npz', **arrays)
    write_json(output/'surface_conversion.json', provenance)
    print(json.dumps(dict(output=str(output/'vehicle.npz'), **stats),ensure_ascii=False,indent=2))


def self_check(output=None):
    prepared = dict(actor_center_m=[1.2,-.4,.7], metres_per_texture_unit=5.1,
                    axis_permutation=[0,2,1], reflection_winding_reversed=True)
    receipt = dict(status='completed', official_mesh_normalization=dict(
        center_in_input_coordinates=[.1,.2,-.3], scale_input_to_output=1.234,
        internal_axis_rotation='[x,y,z] -> [x,-z,y], undone by postprocess_mesh'))
    contract = coordinate_contract(prepared, receipt)
    vertices_actor = np.asarray([[0.,0.,0.],[2.,0.,0.],[0.,1.,0.]])
    vertices_output = actor_to_output(vertices_actor, contract)
    faces = np.asarray([[0,2,1]], np.int64)
    uv = np.asarray([[0.,0.],[1.,0.],[0.,1.]])
    texture = np.asarray([[[255,0,0,0],[0,255,0,85]],[[0,0,255,170],[255,255,255,255]]], np.uint8)
    probes = np.asarray([[0,0],[1,0],[0,1],[1,1],[.25,.75]])
    expected = np.asarray([[0,0,1,2/3],[1,1,1,1],[1,0,0,0],[0,1,0,1/3],[.625,.25,.25,.25]])
    observed = bilinear_rgba(texture, probes)
    assert np.allclose(observed, expected, atol=1e-14)
    arrays, stats = sample_surface(vertices_output, faces, uv, texture, contract, count=256, seed=SEED)
    bary = arrays['surface_barycentric'].astype(np.float64)
    reconstructed = np.einsum('ni,nij->nj',bary,vertices_actor[faces[arrays['surface_face_id']]])
    assert abs(stats['surface_area_m2']-1.) < 1e-12
    assert stats['sampling']['actual_count']==256 and stats['sampling']['exact_requested_count']
    assert len(np.unique(arrays['xyz'],axis=0))==256
    assert stats['sampling']['nearest_neighbor_distance_m']['p00'] > 0
    assert np.allclose(output_to_actor(vertices_output,contract),vertices_actor,atol=1e-12)
    assert np.max(np.abs(reconstructed-arrays['xyz'])) < 2e-7
    assert np.allclose(bary.sum(1),1,atol=1e-7) and (bary>=0).all()
    assert np.allclose(arrays['xyz'][:,2],0,atol=1e-7)
    assert np.allclose(arrays['surface_normal_actor'],[0,0,1],atol=1e-7)
    assert np.allclose(arrays['scales'],[.6/16,.6/16,.1/16],atol=1e-8)
    assert np.allclose(arrays['coefficients'][:,0]*C0+.5,arrays['surface_rgb_generated_srgb'],atol=1e-7)
    assert (arrays['coefficients'][:,1:]==0).all()
    assert np.all(arrays['opacity']==np.float32(.9))
    assert np.ptp(arrays['surface_basecolor_alpha']) > .1
    assert not arrays['is_observed'].any() and (arrays['source_raw']==-1).all()
    normals=np.asarray([[0,0,1],[0,0,-1],[1,0,0],[0,1,0],[1,2,3]],float)
    normals/=np.linalg.norm(normals,axis=1)[:,None]
    q=normal_quaternions(normals)
    matrix=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
    assert np.allclose(matrix[:,:,2],normals,atol=1e-12)
    assert np.allclose(np.linalg.det(matrix),1,atol=1e-12)
    # proper Rz(pi) 只修正 actor 航向；纹理坐标和来源保持原有定义。
    rz_pi=np.diag([-1.,-1.,1.,1.])
    fixed_prepared=dict(prepared,post_actor_transform=rz_pi.tolist(),axis_heading_fix='Rz(pi): semantic front aligned to actor +X')
    fixed_contract=coordinate_contract(fixed_prepared,receipt)
    assert np.allclose(fixed_contract['actor_from_output'],rz_pi@contract['actor_from_output'],atol=1e-14)
    assert np.allclose(output_to_actor(vertices_output,fixed_contract),apply_transform(vertices_actor,rz_pi),atol=1e-12)
    assert np.allclose(actor_to_output(output_to_actor(vertices_output,fixed_contract),fixed_contract),vertices_output,atol=1e-12)
    oblique=np.asarray([[0.,0.,0.],[2.,0.,1.],[0.,1.,.3]])
    oblique_output=actor_to_output(oblique,contract)
    before_fix,_=sample_surface(oblique_output,faces,uv,texture,contract,count=32,seed=SEED)
    after_fix,post_stats=sample_surface(oblique_output,faces,uv,texture,fixed_contract,count=32,seed=SEED)
    assert np.allclose(after_fix['xyz'],apply_transform(before_fix['xyz'],rz_pi),atol=2e-7)
    assert np.allclose(after_fix['surface_normal_actor'],before_fix['surface_normal_actor']@rz_pi[:3,:3].T,atol=1e-7)
    assert np.array_equal(after_fix['surface_face_id'],before_fix['surface_face_id'])
    assert np.allclose(after_fix['surface_barycentric'],before_fix['surface_barycentric'],atol=1e-7)
    assert np.allclose(after_fix['surface_uv'],before_fix['surface_uv'],atol=1e-7)
    assert np.allclose(after_fix['surface_rgb_generated_srgb'],before_fix['surface_rgb_generated_srgb'],atol=1e-7)
    assert not contract['axis_heading_fix']['applied'] and fixed_contract['axis_heading_fix']['applied']
    # 同 seed 的面积采样逐点可复现；仅验证小合成面，不读取任何候选网格。
    repeated,_=sample_surface(vertices_output,faces,uv,texture,contract,count=256,seed=SEED)
    assert np.array_equal(arrays['xyz'],repeated['xyz'])
    # 单个解析三角面核对正式行数，仍不读取候选网格或运行学生。
    dense_points,_,_,dense_sampling=even_area_samples(vertices_actor,faces,np.asarray([True]),SAMPLE_COUNT,SEED,1.)
    assert len(dense_points)==SAMPLE_COUNT and len(np.unique(dense_points,axis=0))==SAMPLE_COUNT
    grid=np.c_[np.asarray(list(__import__('itertools').product(range(5),repeat=2)),float),np.zeros(25)]
    farthest_ids=farthest_subset(grid,9)
    assert len(np.unique(farthest_ids))==9 and np.array_equal(farthest_ids,farthest_subset(grid,9))
    with tempfile.TemporaryDirectory(prefix='vehicle-surface-math-') as folder:
        path=Path(folder)/'synthetic_vehicle.npz'
        np.savez_compressed(path,**arrays)
        with np.load(path,allow_pickle=False) as loaded:
            assert loaded['coefficients'].shape==(256,16,3)
            assert np.array_equal(loaded['surface_face_id'],arrays['surface_face_id'])
    result=dict(passed=True,synthetic_only=True,formal_candidate_mesh_read=False,
        checks=['two-stage nonzero-center normalization roundtrip','proper post-actor Rz(pi) composition, inverse, oblique normals, stable UV/RGB',
                'triangle area and barycentric reconstruction',
                'near-uniform surface sampler actual unique count and nearest-neighbor spacing',
                'reflected-axis winding and normal','RGBA corner/center analytic color and alpha',
                'wxyz thin-axis quaternion incl opposite normals','linear metric sigma and fixed opacity independent of texture alpha',
                'sRGB SH0 reconstruction and zero higher degrees','deterministic seed and Vehicle NPZ roundtrip'],
        production_count=SAMPLE_COUNT,production_seed=SEED,synthetic_samples=256,
        production_count_single_triangle_check=dense_sampling,
        sampling=stats['sampling'],renderer_alpha_coverage_checked=False,
        normalization_roundtrip_max_abs=stats['mesh_normalization_roundtrip_max_abs'],
        post_actor_rz_pi_roundtrip_max_abs=post_stats['mesh_normalization_roundtrip_max_abs'],
        thin_axis_max_l2=stats['quaternion_thin_axis_max_l2'],gpu_calls=0,training_updates=0)
    if output:
        output=Path(output)
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True,exist_ok=True)
        write_json(output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh',type=Path)
    parser.add_argument('--basecolor',type=Path)
    parser.add_argument('--transform',type=Path)
    parser.add_argument('--texture-receipt',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--pbr-map',action='append',default=[],help='role=/absolute/path; retain independent PBR map')
    parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--check-output',type=Path)
    args=parser.parse_args()
    if args.self_check:
        if any(v is not None for v in [args.mesh,args.basecolor,args.transform,args.texture_receipt,args.output]):
            parser.error('--self-check accepts no candidate mesh or production output')
        self_check(args.check_output)
        return
    if not all([args.mesh,args.basecolor,args.transform,args.texture_receipt,args.output]):
        parser.error('production requires --mesh --basecolor --transform --texture-receipt --output')
    convert(args.mesh,args.basecolor,args.transform,args.texture_receipt,args.output,args.pbr_map)


if __name__=='__main__':
    main()
