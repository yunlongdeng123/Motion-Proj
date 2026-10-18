"""从独立输入加载原静态分支和新刚体实例；旧模型/输入只读。"""
import json
import os
from pathlib import Path
import shutil
import sys


def repair_policy(spec):
    policy = spec.get('background_repair')
    if policy is None:
        return None
    if (policy.get('schema') != 'preserve_joint_background_v2'
            or policy.get('initialization') != 'preserve_all_GN11_static_Gaussians'
            or policy.get('freeze_environment') is not True
            or type(policy.get('learning_rate_scale')) not in (int, float)
            or not 0 < policy['learning_rate_scale'] <= 1):
        raise ValueError('invalid background repair policy')
    return policy


def initialize_background(static, instances, spec):
    policy = repair_policy(spec)
    if policy is None:
        raise ValueError('fresh repair runs require an explicit preserve_joint_background_v2 policy; use the preserved historical campaign only for legacy replay')
    count = len(static.get_xyz)
    if count != policy['expected_initial_points']:
        raise ValueError('GN11 background point count differs from bound repair prior')
    return {'criterion': 'preserve_all_GN11_static_Gaussians_no_center_box_deletion',
            'removed_points': 0, 'remaining_points': count,
            'background_joint_optimization': True,
            'shadow_or_Gaussian_extent_cleanup_verified': False}


def validate_spec(path):
    spec = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    repair_policy(spec)
    if not isinstance(spec.get('scene_id'), str) or not spec['scene_id']:
        raise ValueError('需要明确scene_id')
    for key in ('source_repository', 'source_scene', 'prepared', 'reference_config', 'background_checkpoint', 'environment_checkpoint', 'matrix_tools'):
        if key not in spec or not Path(spec[key]).exists():
            raise ValueError('真实场景绑定缺失: ' + key)
    prepared = Path(spec['prepared'])
    report = json.loads((prepared / 'preparation_report.json').read_text(encoding='utf-8-sig'))
    if not report['geometry_ready'] or len(report['frames']) != 49:
        raise ValueError('需要真实49帧刚体实例几何准备')
    import numpy as np
    with np.load(prepared / 'instances.npz', allow_pickle=False) as bundle:
        required = ('local_xyz', 'rgb', 'scale', 'point_ids', 'instance_ids', 'timestamps', 'frame_ids', 'poses', 'valid', 'box_size', 'normalization', 'mask_known')
        if any(key not in bundle for key in required):
            raise ValueError('真实刚体bundle字段不完整')
        arrays = {key: bundle[key] for key in required}
        instance_ids, frame_ids = arrays['instance_ids'], arrays['frame_ids']
        if (instance_ids.ndim != 1 or len(instance_ids) == 0 or instance_ids.dtype.kind != 'U'
                or any(not value for value in instance_ids) or len(set(instance_ids.tolist())) != len(instance_ids)):
            raise ValueError('instance_ids需要非空唯一真实Unicode身份')
        k = len(instance_ids)
        if (frame_ids.shape != (49,) or frame_ids.dtype.kind != 'U' or any(not value for value in frame_ids)
                or len(set(frame_ids.tolist())) != 49):
            raise ValueError('frame_ids需要49个非空唯一身份')
        if frame_ids.tolist() != [frame.get('frame_id') for frame in report['frames']]:
            raise ValueError('逐帧身份与准备report不符')
        xyz, rgb, scales = arrays['local_xyz'], arrays['rgb'], arrays['scale']
        if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0 or rgb.shape != xyz.shape or scales.shape != xyz.shape:
            raise ValueError('local_xyz/rgb/scale需要一致非空Nx3形状')
        if any(value.dtype.kind not in 'fiu' or not np.isfinite(value).all() for value in (xyz, rgb, scales)):
            raise ValueError('local_xyz/rgb/scale需要finite实数')
        if (rgb < 0).any() or (rgb > 1).any() or (scales <= 0).any():
            raise ValueError('rgb需要在0..1内且scale必须positive')
        point_ids, times = arrays['point_ids'], arrays['timestamps']
        if (point_ids.shape != (len(xyz),) or point_ids.dtype.kind not in 'iu'
                or (point_ids < 0).any() or (point_ids >= k).any()):
            raise ValueError('point_ids需要合法Nx1整数实例索引')
        if (times.shape != (49,) or times.dtype.kind not in 'fiu' or not np.isfinite(times).all()
                or (np.diff(times.astype(np.float64)) <= 2e-6).any()):
            raise ValueError('timestamps需要49个finite严格递增且可区分的模型时间')
        poses, valid, sizes, known = (arrays[name] for name in ('poses', 'valid', 'box_size', 'mask_known'))
        if poses.shape != (49, k, 4, 4) or valid.shape != (49, k) or sizes.shape != (49, k, 3) or known.shape != (49, k):
            raise ValueError('实例pose/valid/box_size/mask_known形状不符')
        if valid.dtype.kind != 'b' or known.dtype.kind != 'b' or not valid.any(axis=0).all():
            raise ValueError('valid/mask_known需要明确bool且每条轨迹至少有一个真实观测')
        observed, observed_sizes = poses[valid], sizes[valid]
        if (poses.dtype.kind not in 'fiu' or sizes.dtype.kind not in 'fiu'
                or not np.isfinite(observed).all() or not np.isfinite(observed_sizes).all() or (observed_sizes <= 0).any()):
            raise ValueError('有效pose必须finite且真实box_size必须finite positive')
        rotations = observed[:, :3, :3]
        if (not np.allclose(rotations.swapaxes(-1, -2) @ rotations, np.eye(3), atol=2e-4, rtol=2e-4)
                or not np.allclose(np.linalg.det(rotations), 1, atol=2e-4, rtol=2e-4)
                or not np.allclose(observed[:, 3], [0, 0, 0, 1], atol=1e-6, rtol=0)):
            raise ValueError('有效pose必须为proper刚体齐次变换')
        normalization = arrays['normalization']
        if normalization.shape != (4, 4) or normalization.dtype.kind not in 'fiu' or not np.isfinite(normalization).all():
            raise ValueError('真实场景归一化缺失')
        if report.get('instance_count') != k or report.get('real_instance_points') != len(xyz):
            raise ValueError('实例/真实点数与准备report不符')
        if report.get('mask_known_pairs') != int(known.sum()):
            raise ValueError('mask_known实际sum与准备report不符')
        if bool(known.all()) != report['full_mask_supervision_ready']:
            raise ValueError('mask就绪receipt与实际bundle不符')
    return spec, report


def make_sidecar(spec, output):
    """只在新目录写 loader 的 transform 等副产物；只读大资产使用软链接。"""
    source = Path(spec['source_scene']).resolve()
    sidecar = Path(output) / 'input_sidecar'
    sidecar.mkdir()
    for name in ('image', 'calib', 'pose', 'velodyne', 'sky', 'timestamp'):
        os.symlink(source / name, sidecar / name, target_is_directory=True)
    for name in ('points3d.ply',):
        os.symlink(source / name, sidecar / name)
    return sidecar


def load_models(spec, output, *, freeze_background=True, restore=None):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    if __package__:
        from .rigid_model import RigidInstanceGaussianModel
    else:
        from rigid_model import RigidInstanceGaussianModel
    source = Path(spec['source_repository']).resolve()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    from scene import Scene, GaussianModel, EnvLight
    config = json.loads(Path(spec['reference_config']).read_text(encoding='utf-8-sig'))
    config.update(source_path=str(make_sidecar(spec, output)), model_path=str(output),
                  including_dynamic=False, adding=False, training=True, eval=False)
    args = OmegaConf.create(config)
    static = GaussianModel(args)
    scene = Scene(args, static, None, shuffle=False)
    cameras = scene.getTrainCameras()
    prepared = np.load(Path(spec['prepared']) / 'instances.npz', allow_pickle=False)
    times = prepared['timestamps']
    if len(cameras) != 49 or not np.allclose([c.timestamp for c in cameras], times, atol=1e-7, rtol=0):
        raise ValueError('真实相机49时间与实例轨迹不匹配')
    if [Path(c.image_name).stem for c in cameras] != list(prepared['frame_ids']):
        raise ValueError('真实相机身份与prepared逐帧身份不匹配')
    recomputed = np.loadtxt(Path(config['source_path']) / 'transform_matrix.txt')
    if not np.allclose(recomputed, prepared['normalization'], atol=1e-8, rtol=0):
        raise ValueError('原静态/camera和新实例归一化坐标不匹配')
    # map_location=cuda 恢复当前设备。背景保留原局部模型和Adam，随后删除选中车辆
    # 的盒内静态高斯，避免移动停着的车时出现原静态车体副本。
    if restore is None:
        static_args, step = torch.load(spec['background_checkpoint'], map_location='cuda', weights_only=False)
        if step != 30000:
            raise ValueError('静态先验必须是真实30k模型')
        static.restore(static_args, args)
        instances = RigidInstanceGaussianModel.from_npz(Path(spec['prepared']) / 'instances.npz', device='cuda', sh_degree=args.sh_degree)
        dropped = initialize_background(static, instances, spec)
        if repair_policy(spec) is not None:
            # Keep pretrained Gaussian values, start Adam for the new objective.
            # On resume the saved repair Adam is restored instead.
            static.training_setup(args)
            for group in static.optimizer.param_groups:
                group['lr'] *= repair_policy(spec)['learning_rate_scale']
        rigid_optimizer = instances.make_optimizer({'xyz': args.position_lr_init, 'f_dc': args.feature_lr,
            'f_rest': args.feature_lr / 20, 'opacity': args.opacity_lr, 'scaling': args.scaling_lr,
            'rotation': args.rotation_lr}, eps=1e-15)
    else:
        static.restore(restore['background'], args)
        instances = RigidInstanceGaussianModel.from_capture(restore['instances'], device='cuda')
        rigid_optimizer = instances.optimizer
        dropped = restore['background_exclusion']
    environment = EnvLight(args.env_map_res).cuda()
    if restore is None:
        captured, env_step = torch.load(spec['environment_checkpoint'], map_location='cuda', weights_only=False)
        if env_step != 30000:
            raise ValueError('环境先验必须是真实30k模型')
        environment.restore(captured, args)
    else:
        environment.restore(restore['environment'], args)
    set_background_trainable(static, environment, not freeze_background)
    if repair_policy(spec) is not None:
        # Evaluation preserves the checkpoint's trainability flags for exact
        # model/Adam restore verification; rendering remains inside no_grad.
        set_background_trainable(static, environment, True)
        for parameter in environment.parameters():
            parameter.requires_grad_(False)
    return args, scene, static, instances, rigid_optimizer, environment, dropped


def set_background_trainable(static, environment, enabled):
    for group in static.optimizer.param_groups:
        for parameter in group['params']:
            parameter.requires_grad_(enabled)
    for parameter in environment.parameters():
        parameter.requires_grad_(enabled)


def remove_vehicle_background(static, instances, chunk_size=65536):
    """按每个实际时刻的刚体盒删除原静态车体；不只处理移动实例。

    与原静态 PVG 的当前 SHM 位置比较。中心盒判据不能保证消除跨界Gaussian
    的尾部及真实投影阴影，必须在后续删除渲染中单独检查背景残留。
    """
    import torch
    remove = torch.zeros(len(static.get_xyz), dtype=torch.bool, device=static.get_xyz.device)
    with torch.no_grad():
        for frame, timestamp in enumerate(instances.timestamps):
            xyz = static.get_xyz_SHM(float(timestamp))
            poses = instances.base_poses[frame]
            for instance in torch.nonzero(instances.pose_valid[frame], as_tuple=False).flatten():
                pose = poses[instance]
                half = instances.box_size[frame, instance] / 2
                if bool((half <= 0).any()):
                    raise ValueError('有效实例缺少真实尺寸')
                # 先世界AABB排除；仅候选点计算精确OBB。
                extent = pose[:3, :3].abs() @ half
                broad = (~remove) & ((xyz - pose[:3, 3]).abs() <= extent).all(1)
                indices = torch.nonzero(broad, as_tuple=False).flatten()
                for ids in indices.split(chunk_size):
                    local = (xyz[ids] - pose[:3, 3]) @ pose[:3, :3]
                    remove[ids[(local.abs() <= half).all(1)]] = True
        count = int(remove.sum())
        static.prune_points(remove)
    return {'criterion': 'any_valid_frame_Gaussian_SHM_center_in_exact_instance_box',
            'removed_points': count, 'remaining_points': len(static.get_xyz),
            'stationary_included': True, 'shadow_or_Gaussian_extent_cleanup_verified': False}


def masks_for_frame(spec, model, camera):
    import numpy as np
    import torch
    from PIL import Image
    frame_id = Path(camera.image_name).stem
    with np.load(Path(spec['prepared']) / 'instances.npz', allow_pickle=False) as bundle:
        row = list(bundle['frame_ids']).index(frame_id)
        known = bundle['mask_known'][row]
    height, width = camera.image_height, camera.image_width
    masks = []
    path = Path(spec['prepared']) / 'instance_masks' / (frame_id + '.png')
    if known.any() and not path.is_file():
        raise FileNotFoundError('known实例mask实际文件缺失: ' + str(path))
    if path.is_file():
        image = Image.open(path)
        labels = np.array(image.resize((width, height), Image.Resampling.NEAREST))
    else:
        labels = np.zeros((height, width), dtype=np.uint16)
    for instance, track in enumerate(model.instance_ids):
        masks.append(torch.from_numpy((labels == instance + 1).astype(np.float32)))
    return torch.stack(masks).to(model.local_xyz.device), torch.as_tensor(known, device=model.local_xyz.device)
