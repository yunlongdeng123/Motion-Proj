"""只在新目录写入真实训练输入；不读取诊断帧 LiDAR，不调用 GPU。"""
from pathlib import Path
import itertools
import json
import math
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/vehicle_completion_prior_20261122_r1/conditioning')
BANK = Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs')
MASK = BANK.parent / 'repaired_masks_all_r4'
NATIVE = Path('/root/autodl-tmp/drivingeditor_repro/campaign/waymo_val065_allframes10k_20261109_r1/inputs/native')
TRACK = 'bzFbWRrSGOgEaXxnuGV0tw'
OCCLUDER = 'NTjq-OhZ_97OMJ4xwJAxXg'
TRAIN = [86, 110]
DIAGNOSTIC = [134, 179, 198]
VOXEL_M = 0.02


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def mat(value):
    return np.asarray(value, dtype=np.float64).reshape(4, 4)


def transform(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def brown_project(world, obs):
    camera = mat(obs['camera_pose_world']) @ mat(obs['camera_extrinsic'])
    pc = transform(world, np.linalg.inv(camera))
    assert (pc[:, 0] > 0).all(), '目标 OBB 必须完全位于正深度'
    x, y = -pc[:, 1] / pc[:, 0], -pc[:, 2] / pc[:, 0]
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = obs['camera_intrinsic_Brown']
    r2 = x*x + y*y
    radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
    return np.c_[fx*(x*radial + 2*p1*x*y + p2*(r2+2*x*x)) + cx,
                 fy*(y*radial + p1*(r2+2*y*y) + 2*p2*x*y) + cy], pc


def hull(points):
    values = sorted(set(map(tuple, points.tolist())))
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in values:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(values):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def box_projection(obs, box):
    corners = np.asarray(list(itertools.product([-1, 1], repeat=3)), float) * np.asarray(box['dimensions']) / 2
    samples = [corners]
    for i, a in enumerate(corners):
        for b in corners[i+1:]:
            if np.count_nonzero(a != b) == 1:
                t = np.linspace(0, 1, 33)[:, None]
                samples.append(a[None, :]*(1-t)+b[None, :]*t)
    world = transform(np.concatenate(samples), mat(box['world_from_actor']))
    uv, _ = brown_project(world, obs)
    polygon = hull(uv)
    corner_uv, _ = brown_project(transform(corners, mat(box['world_from_actor'])), obs)
    return polygon, [float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max())], corner_uv


def mask_image(value):
    return Image.fromarray(value.astype(np.uint8)*255, 'L')


def save_mask(path, value):
    mask_image(value).save(path)


def process_frame(raw, role):
    out = ROOT / role / f'{raw:03d}'
    out.mkdir(parents=True, exist_ok=False)
    obs_path = BANK / 'observations' / f'{raw:03d}.json'
    obs = json.loads(obs_path.read_text())
    box = next(b for b in obs['boxes'] if b['track'] == TRACK)
    actor = mat(box['world_from_actor'])
    camera = mat(obs['camera_pose_world']) @ mat(obs['camera_extrinsic'])
    rgb_path = NATIVE / obs['image_path']
    rgb = np.asarray(Image.open(rgb_path).convert('RGB'))
    w, h = obs['camera_size']
    assert rgb.shape == (h, w, 3)
    mask_dir = MASK / f'{raw:03d}'
    a20 = np.asarray(Image.open(mask_dir/'a20.png')) > 0
    a9 = np.asarray(Image.open(mask_dir/'a9.png')) > 0
    labels = np.asarray(Image.open(mask_dir/'labels.png'))
    assert a20.shape == a9.shape == labels.shape == (320, 480)
    assert not (a20 & a9).any()
    assert (labels[a20] == 21).all() and (labels[a9] == 10).all()
    # 原生 480 掩码有近似边界；核心收缩一格，a9 边界向外隔离一格。
    target_core = np.asarray(mask_image(a20 & (labels == 21)).filter(ImageFilter.MinFilter(3))) > 0
    a9_guard = np.asarray(mask_image(a9 | (labels == 10)).filter(ImageFilter.MaxFilter(3))) > 0
    safe480 = target_core & ~a9_guard
    resize = lambda x: np.asarray(mask_image(x).resize((w, h), Image.Resampling.NEAREST)) > 0
    approx, safe, occ, occ_guard = map(resize, [a20, safe480, a9, a9_guard])
    polygon, bbox, corner_uv = box_projection(obs, box)
    potential_img = Image.new('L', (w, h), 0)
    ImageDraw.Draw(potential_img).polygon(polygon, fill=255)
    # OBB 仅提供潜在范围；保守加 4px native guard，并覆盖标签观测边界。
    potential = np.asarray(potential_img.filter(ImageFilter.MaxFilter(9))) > 0
    potential |= approx
    unknown = potential & ~safe
    background = ~potential
    tri = np.zeros((h, w), np.uint8)
    tri[unknown] = 128
    tri[safe] = 255
    assert not (safe & occ_guard).any()
    assert np.all(unknown | safe | background)
    assert not (unknown & safe).any()
    pad_x, pad_y = max(40, int((bbox[2]-bbox[0])*.12)), max(32, int((bbox[3]-bbox[1])*.12))
    crop = [max(0, math.floor((bbox[0]-pad_x)/4)*4), max(0, math.floor((bbox[1]-pad_y)/4)*4),
            min(w, math.ceil((bbox[2]+pad_x)/4)*4), min(h, math.ceil((bbox[3]+pad_y)/4)*4)]
    x0, y0, x1, y1 = crop
    s = np.s_[y0:y1, x0:x1]
    partial = rgb.copy()
    partial[~safe] = 0
    rgba = np.dstack([partial, safe.astype(np.uint8)*255])
    Image.fromarray(rgb[s], 'RGB').save(out/'context_rgb.png')
    Image.fromarray(partial[s], 'RGB').save(out/'partial_visible_rgb.png')
    Image.fromarray(rgba[s], 'RGBA').save(out/'partial_visible_rgba.png')
    for name, value in [('visible_mask',safe), ('visible_mask_approx',approx), ('occluder_mask',occ),
                        ('occluder_guard_mask',occ_guard), ('unknown_potential_mask',unknown),
                        ('amodal_box_potential_mask',potential), ('known_background_mask',background)]:
        save_mask(out/f'{name}.png', value[s])
    Image.fromarray(tri[s], 'L').save(out/'tri_state_mask.png')
    sc = np.s_[y0//4:y1//4, x0//4:x1//4]
    Image.fromarray(labels[sc].astype(np.uint16)).save(out/'labels_native480_crop.png')
    save_mask(out/'a20_mask_native480_crop.png', a20[sc])
    save_mask(out/'a9_mask_native480_crop.png', a9[sc])
    stored = np.asarray([p for p in box['corners2D'] if p is not None])
    stored_aabb = np.r_[stored.min(0), stored.max(0)]*2
    projected_corner_aabb = np.r_[corner_uv.min(0), corner_uv.max(0)]
    mapping = dict(schema='a20.real_partial_conditioning.v1', raw=raw, role=role, track=TRACK,
        alias='a20', native_instance_label=21, identity='black pickup truck',
        occluder=dict(alias='a9', native_instance_label=10, track=OCCLUDER, identity='white Mazda'),
        dimensions_xyz_m=box['dimensions'], canonical_axes='actor x=longitudinal forward, y=left, z=up; centered OBB; metres',
        image_native_size=[w,h], original_mask_size=[480,320], crop_native_xyxy_half_open=crop,
        crop_size=[x1-x0,y1-y0], crop_to_native_homography=[[1,0,x0],[0,1,y0],[0,0,1]],
        native_to_crop_homography=[[1,0,-x0],[0,1,-y0],[0,0,1]],
        mask480_to_native_pixel_convention='continuous u_native=4*u_mask; pixel centers u_native=4*(u_mask+0.5); nearest mask expansion, no new boundary truth',
        source_mask_crop480_xyxy_half_open=[x0//4,y0//4,x1//4,y1//4],
        camera_intrinsic_Brown_native=obs['camera_intrinsic_Brown'],
        camera_intrinsic_Brown_crop=[obs['camera_intrinsic_Brown'][0],obs['camera_intrinsic_Brown'][1],obs['camera_intrinsic_Brown'][2]-x0,obs['camera_intrinsic_Brown'][3]-y0,*obs['camera_intrinsic_Brown'][4:]],
        camera_axes='Waymo camera native X forward, Y left, Z up; normalized image x=-Y/X, y=-Z/X',
        camera_to_world=camera.tolist(), world_from_actor=actor.tolist(),
        camera_from_actor=(np.linalg.inv(camera)@actor).tolist(),
        opencv_camera_from_waymo_camera=[[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]],
        ego_to_world=mat(obs['ego_world']).tolist(),
        camera_position_canonical_m=transform(camera[None,:3,3],np.linalg.inv(actor))[0].tolist(),
        projected_obb_bbox_native=bbox, projected_obb_convex_hull_native=polygon,
        stored_box_native_projection_aabb_max_error_px=float(abs(stored_aabb-projected_corner_aabb).max()),
        mask_contract=dict(tri_state_values={'0':'outside fixed OBB potential / target background','128':'unknown within potential, including occluded and uncertain edges','255':'conservative observed target core'},
            potential='Brown projected fixed OBB, 33 samples per edge, convex hull + 4 native pixel guard; union observed approximate mask. This is not amodal silhouette GT.',
            safe_visible='a20 and native21 agreement; erode one 480-mask pixel; exclude one-pixel dilated a9/native10; RGB/alpha outside core forced to zero',
            occluder='a9 mask within crop; its overlap with target potential remains unknown',
            resolution='480x320 source masks are approximate; 1920x1280 masks are nearest expansion, not native-resolution annotation'),
        source=dict(observation=str(obs_path),rgb=str(rgb_path),target_mask=str(mask_dir/'a20.png'),occluder_mask=str(mask_dir/'a9.png'),labels=str(mask_dir/'labels.png')),
        counts=dict(visible_source480=int(a20.sum()),visible_core_source480=int(safe480.sum()),
            visible_native=int(safe.sum()),visible_approx_native=int(approx.sum()),
            occluder_crop_native=int(occ[s].sum()),potential_crop_native=int(potential[s].sum()),
            unknown_potential_crop_native=int(unknown[s].sum()),
            known_background_crop_native=int(background[s].sum()),
            occluded_unknown_crop_native=int((unknown & occ)[s].sum()),
            safe_visible_intersect_occluder_or_guard=0))
    write_json(out/'mapping.json',mapping)
    return obs, box, mapping


def build_lidar(records):
    all_xyz, all_raw, all_rows, stats, transforms = [], [], [], [], []
    for raw in TRAIN:
        obs, box, _ = records[raw]
        source_path = BANK / obs['lidar_path']
        assert raw in TRAIN and raw not in DIAGNOSTIC
        cloud = np.fromfile(source_path,dtype='<f4').reshape(-1,6)
        assert np.isfinite(cloud).all()
        ego = mat(obs['ego_world']); actor = mat(box['world_from_actor'])
        actor_from_ego = np.linalg.inv(actor) @ ego
        local = transform(cloud[:,:3].astype(np.float64),actor_from_ego)
        inside = (np.abs(local) <= np.asarray(box['dimensions'])/2 + 1e-6).all(1)
        # 与 a9 扩张 OBB 相交的点保守移除；不以 camera mask 删除后方 LiDAR 可见点。
        a9 = next(b for b in obs['boxes'] if b['track'] == OCCLUDER)
        a9_local = transform(cloud[:,:3].astype(np.float64), np.linalg.inv(mat(a9['world_from_actor']))@ego)
        in_a9_guard = (np.abs(a9_local) <= np.asarray(a9['dimensions'])/2 + 0.05).all(1)
        keep = inside & ~in_a9_guard
        source_rows = np.flatnonzero(keep)
        xyz = local[keep]
        all_xyz.append(xyz)
        all_raw.append(np.full(len(xyz),raw,np.int16))
        all_rows.append(source_rows.astype(np.int64))
        transforms.append(actor_from_ego)
        stats.append(dict(raw=raw,source_path=str(source_path),source_rows=len(cloud),
            target_obb_points_before_a9_guard=int(inside.sum()),
            metadata_measured_lidar_points=int(box['measured_lidar_points']),
            excluded_a9_obb_guard_points=int((inside & in_a9_guard).sum()),kept_measured_points=len(xyz),
            canonical_min_m=xyz.min(0).tolist(),canonical_max_m=xyz.max(0).tolist(),
            a9_center_canonical_m=transform(np.asarray(a9['center_world'])[None],np.linalg.inv(actor))[0].tolist()))
        assert int(inside.sum()) == box['measured_lidar_points']
    points = np.concatenate(all_xyz)
    source_raw = np.concatenate(all_raw)
    source_rows = np.concatenate(all_rows)
    voxel_coords = np.floor(points/VOXEL_M).astype(np.int64)
    _, first, inverse, counts = np.unique(voxel_coords,axis=0,return_index=True,return_inverse=True,return_counts=True)
    # 每格选距离该格真实测量均值最近的真实点；不合成或补点。
    means = np.zeros((len(first),3),np.float64)
    np.add.at(means,inverse,points)
    means /= counts[:,None]
    representatives = np.empty(len(first),np.int64)
    membership = np.zeros((len(first),len(TRAIN)),bool)
    for v in range(len(first)):
        ids=np.flatnonzero(inverse==v)
        representatives[v]=ids[np.argmin(np.sum((points[ids]-means[v])**2,axis=1))]
        for j,raw in enumerate(TRAIN):membership[v,j]=np.any(source_raw[ids]==raw)
    dedup=points[representatives]
    lidar_dir=ROOT/'lidar'
    lidar_dir.mkdir(exist_ok=False)
    np.savez_compressed(lidar_dir/'aggregated_lidar_train_086_110.npz',
        xyz_canonical_m=dedup.astype(np.float32),observed_points_actor_native_m=dedup.astype(np.float32),
        size_lwh_m=np.asarray([5.0,2.5,2.08],np.float32),source_raw=source_raw[representatives],
        measured_lidar_only=np.bool_(True),native_gaussian_control_used=np.bool_(False),
        train_source_raw_ids=np.asarray(TRAIN,np.int16),
        source_frame_membership=membership,representative_source_raw_id=source_raw[representatives],
        representative_source_row=source_rows[representatives],voxel_size_m=np.float32(VOXEL_M),
        voxel_member_counts=counts.astype(np.int32),all_xyz_canonical_m=points.astype(np.float32),
        all_source_raw_id=source_raw,all_source_row=source_rows,all_to_voxel_index=inverse.astype(np.int32),
        actor_from_ego_by_train_raw=np.asarray(transforms),dimensions_xyz_m=np.asarray([5.0,2.5,2.08],np.float32))
    with (lidar_dir/'aggregated_lidar_train_086_110.ply').open('w') as stream:
        stream.write(f'ply\nformat ascii 1.0\nelement vertex {len(dedup)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n')
        for point in dedup:stream.write(' '.join(f'{v:.7f}' for v in point)+'\n')
    result=dict(schema='a20.train_measured_lidar.v1',train_source_raw_ids=TRAIN,diagnostic_lidar_read=[],
        measured_lidar_only=True,native_gaussian_control_used=False,
        source_raw_key='source_raw',size_lwh_m=[5.0,2.5,2.08],
        npz='aggregated_lidar_train_086_110.npz',primary_key='xyz_canonical_m',units='metres',
        axes='right-handed actor local: x=vehicle longitudinal forward, y=left, z=up; centered fixed OBB',
        dimensions_xyz_m=[5.0,2.5,2.08],source_lidar_format='<f4 N x 6; first three columns are same-frame ego xyz metres; other source columns are preserved only by row index',
        transform='p_actor = inverse(world_from_actor) @ ego_world @ p_ego, column-vector SE3; NumPy row-vector implementation equivalent',
        boundary_filter='abs(p_actor)<=dimensions/2+1e-6; subtract a9 OBB expanded 0.05m per axis. Geometric membership, not semantic instance ground truth.',
        no_camera_mask_filter='Camera a9 projection does not decide LiDAR ownership: genuine a20 returns may project behind the occluder. a9 3D OBB guard is used.',
        dedup_policy='2cm canonical voxels; representative is an actual measured point nearest per-voxel measured mean; all originals and membership retained',
        original_point_count=len(points),deduplicated_point_count=len(dedup),
        cross_source_voxel_count=int(membership.all(1).sum()),
        canonical_min_m=dedup.min(0).tolist(),canonical_max_m=dedup.max(0).tolist(),per_source=stats)
    write_json(lidar_dir/'provenance.json',result)
    return result


def contact_sheet(records):
    tile_w, tile_h = 460, 390
    sheet=Image.new('RGB',(tile_w*3,tile_h*5),(17,21,28))
    draw=ImageDraw.Draw(sheet)
    for row,raw in enumerate(TRAIN+DIAGNOSTIC):
        _,_,m=records[raw]
        out=ROOT/m['role']/f'{raw:03d}'
        context=Image.open(out/'context_rgb.png').convert('RGB')
        rgba=Image.open(out/'partial_visible_rgba.png').convert('RGBA')
        checker=Image.new('RGB',rgba.size,(45,49,57))
        checker.paste(rgba,mask=rgba.getchannel('A'))
        tri=np.asarray(Image.open(out/'tri_state_mask.png'))
        color=np.zeros((*tri.shape,3),np.uint8);color[tri==0]=[23,30,40];color[tri==128]=[215,155,45];color[tri==255]=[20,210,155]
        for col,img in enumerate([context,checker,Image.fromarray(color)]):
            img.thumbnail((tile_w-12,tile_h-55))
            x=col*tile_w+(tile_w-img.width)//2;y=row*tile_h+48
            sheet.paste(img,(x,y))
        title=f"raw{raw:03d} {'TRAIN' if m['role']=='train' else 'DIAGNOSTIC ONLY'} | mask480 visible={m['counts']['visible_source480']} core={m['counts']['visible_core_source480']}"
        draw.text((8,row*tile_h+7),title,fill='white')
        for col,title in enumerate(['Native RGB context','Measured visible RGB core','Green=visible / amber=unknown / dark=outside OBB']):
            draw.text((col*tile_w+8,row*tile_h+28),title,fill=(205,211,222))
    sheet.save(ROOT/'contact_sheet.jpg',quality=93)


def main():
    if (ROOT/'summary.json').exists():raise RuntimeError('已有完成 packet，拒绝覆盖')
    ROOT.mkdir(parents=True,exist_ok=True)
    records={}
    for raw in TRAIN:
        records[raw]=process_frame(raw,'train')
    lidar=build_lidar(records)
    print('TRAIN_PACKET_READY '+str(ROOT/'lidar'/'aggregated_lidar_train_086_110.npz'),flush=True)
    for raw in DIAGNOSTIC:
        records[raw]=process_frame(raw,'diagnostic')
    contact_sheet(records)
    summary=dict(schema='a20.real_input_packet.v1',task_id='VEHICLE-COMPLETION-PRIOR-20261122-R1',
        build_raw_ids=TRAIN,diagnostic_only_raw_ids=DIAGNOSTIC,holdout_used_for_build=False,
        diagnostic_lidar_reads=[],gpu_calls=0,model_downloads=0,git_operations=0,
        track=TRACK,identity='black pickup truck',native_instance_label=21,
        physical_size_xyz_m=[5.0,2.5,2.08],
        training_contract='Only train/086, train/110 and lidar/ may condition or select a prior; diagnostic/ and diagnostic sheet rows are excluded from BUILD and candidate selection.',
        frames=[records[r][2] for r in TRAIN+DIAGNOSTIC],lidar=lidar,
        files=dict(training_packet=['train/086','train/110','lidar'],contact_sheet='contact_sheet.jpg'),
        failure_ledger_refs=[],failure_ledger_delta='none')
    write_json(ROOT/'summary.json',summary)
    report='''# a20 真实输入 packet\n\n训练输入固定为 raw086/110；raw134/179/198 仅用于诊断，不读取其 LiDAR，也不参与先验约束或选择。\n\n```mermaid\nflowchart LR\n A[raw086/110 原生 RGB + 480 掩码] --> B[保守可见核心 + Brown OBB 投影]\n B --> C[可见 RGB / 遮挡未知 / 盒外背景]\n D[raw086/110 实测 LiDAR + ego pose] --> E[a20 OBB canonical 米制变换]\n E --> F[a9 三维排除 + 2cm 真实代表点]\n C --> G[训练输入 packet]\n F --> G\n H[134/179/198 RGB + mask] --> I[独立诊断]\n```\n\n掩码来源为 480×320，最近邻扩张到原生 1920×1280；可见核心收缩一格并与 a9 膨胀边界隔离。OBB 投影提供潜在范围，不作为物体轮廓真值；范围内除可见核心外全部为 unknown。RGBA 在可见核心外的 RGB/alpha 均置零，context 图保留场景用于坐标核查。\n\n点云保留真实实测点来源；两帧先各自转入目标 actor 坐标，严格 OBB 筛选并排除 a9 的 5cm 扩张 OBB，之后采用 2cm 体素真实代表点去重。边界筛选不等于语义归属真值。完整数值见 summary.json 和 lidar/provenance.json。\n'''
    (ROOT/'README.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(root=str(ROOT),frames=[dict(raw=r,role=records[r][2]['role'],crop=records[r][2]['crop_size'],counts=records[r][2]['counts']) for r in TRAIN+DIAGNOSTIC],lidar=lidar),indent=2))


if __name__=='__main__':main()
