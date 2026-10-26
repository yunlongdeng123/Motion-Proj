"""冻结原生重建/编辑算子，跨场景生成前三维与固定ProPainter输入。

不把scene0008的门控/道路状态迁移为其他场景答案，不训练新②。
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, '/root/autodl-tmp')
import editor_common as common
from editor_common import np, torch, Image, Camera, cpu, save, read, prepare_rasterization, merge_rows, raster
from scipy.ndimage import binary_dilation


class SceneNative(common.Native):
    def __init__(self, run):
        common.B = Path(run)
        # 旧原生基线有完整四件；部分历史run没有actual_scene_binding，直接核对保存相机与loader时间公式。
        binding = common.B / 'actual_scene_binding.json'
        if binding.exists():
            super().__init__()
        else:
            # 父类只在最后读该JSON；临时替换read，不修改任何旧文件。
            original_read = common.read
            def compatible_read(path):
                if Path(path) == binding:
                    cameras = original_read(common.B / 'cameras.json')
                    config = original_read(common.B / 'effective_config.json')
                    half = config['frame_interval'] * (len(cameras) - 1) / 2
                    return {'timestamps': np.linspace(-half, half, len(cameras)).tolist()}
                return original_read(path)
            common.read = compatible_read
            try:
                super().__init__()
            finally:
                common.read = original_read
        self.source = Path(self.config['source_path'])
        self.scale = float(np.linalg.norm(np.loadtxt(self.source / 'transform_matrix.txt')[:3, 0]))
        assert .0001 < self.scale < 100
        assert len(self.records) == len(self.stamps) == 49

    def camera_at(self, index, lateral_m=0.):
        record = self.records[index]
        c2w = np.eye(4)
        c2w[:3, :3] = record['rotation']
        c2w[:3, 3] = record['position']
        # 相机局部右方向，米制输入；不假定所有场景使用相同世界坐标轴。
        c2w[:3, 3] += c2w[:3, 0] * lateral_m * self.scale
        w2c = np.linalg.inv(c2w)
        return Camera(index, w2c[:3, :3].T.copy(), w2c[:3, 3].copy(),
                      cx=record['cx'] / 4, cy=record['cy'] / 4,
                      fx=record['fx'] / 4, fy=record['fy'] / 4,
                      image=torch.zeros(3, 320, 480), image_name=record['img_name'],
                      resolution=(480, 320), timestamp=self.stamps[index])

    def render_operation(self, camera, operation):
        static = prepare_rasterization(camera, self.static, self.args)
        static['means2D'] = torch.zeros_like(static['means3D'])
        static['features'] = torch.cat((static['features'], torch.zeros_like(static['features'][:, :1])), 1)
        kwargs = static
        if operation not in ['delete_all', 'repeat_delete']:
            dynamic = prepare_rasterization(camera, self.dynamic, self.args, dynamic=True)
            dynamic['means2D'] = torch.zeros_like(dynamic['means3D'])
            if operation == 'move_all_modeled':
                delta = cpu(camera.c2w)[:3, 0] * self.scale
                dynamic['means3D'] = dynamic['means3D'] + torch.as_tensor(delta, dtype=torch.float32, device='cuda')
            z = torch.cat((dynamic['means3D'], torch.ones_like(dynamic['means3D'][:, :1])), 1) @ camera.world_view_transform[:, 2:3]
            dynamic['features'] = torch.cat((z, torch.ones_like(z), torch.ones_like(z)), 1)
            kwargs = merge_rows(kwargs, dynamic)
            if operation == 'insert_all_reused':
                clone = {k: (v.clone() if v is not None else None) for k, v in dynamic.items()}
                delta = cpu(camera.c2w)[:3, 0] * 2 * self.scale
                clone['means3D'] += torch.as_tensor(delta, dtype=torch.float32, device='cuda')
                z = torch.cat((clone['means3D'], torch.ones_like(clone['means3D'][:, :1])), 1) @ camera.world_view_transform[:, 2:3]
                clone['features'] = torch.cat((z, torch.ones_like(z), torch.ones_like(z)), 1)
                kwargs = merge_rows(kwargs, clone)
        out = raster(camera, self.args, self.static.active_sh_degree, kwargs, self.bg)
        env = self.env(camera.get_world_directions(False).permute(1, 2, 0)).permute(2, 0, 1)
        out['rgb'] = out['rgb'] + (1 - out['alpha']) * env
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--single-frame', action='store_true')
    args = parser.parse_args()
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)
    assert not (dest / 'summary.json').exists(), '保留完成结果'
    started = time.time()
    n = SceneNative(args.run)
    operations = ['noop', 'delete_all', 'move_all_modeled', 'insert_all_reused', 'ego_shift', 'repeat_delete']
    for operation in operations:
        for sub in ['native_rgb', 'mask', 'float']:
            (dest / operation / sub).mkdir(parents=True, exist_ok=True)
    rows = []
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for index in ([24] if args.single_frame else range(49)):
            name = n.records[index]['img_name']
            raw = int(name[-3:])
            camera = n.camera_at(index)
            baseline = n.render_operation(camera, 'noop')
            # 额外特征不得改变完整重建；和原通路直接浮点比较。
            original = n.render(camera, dynamic=True)
            identity_difference = float((baseline['rgb'] - original['rgb']).abs().max())
            assert identity_difference <= 1e-6, identity_difference
            gt_file = n.source / 'image/image_0' / (name + '.png')
            gt = np.array(Image.open(gt_file).convert('RGB').resize((480, 320), Image.Resampling.LANCZOS)) / 255.
            mask_file = n.source / 'dynamic_masks/dynamic_mask_0' / (name + '.png')
            body = np.array(Image.open(mask_file).convert('L').resize((480, 320), Image.Resampling.NEAREST)) > 0
            union = binary_dilation(body, iterations=2) | (cpu(baseline['extra'])[0] > .05)
            native = cpu(baseline['rgb']).transpose(1, 2, 0)
            mse = (np.clip(native, 0, 1) - gt) ** 2
            noop_metrics = {domain: float(-10 * np.log10(max(float(mse[mask].mean()), 1e-12))) if mask.any() else None
                            for domain, mask in [('full', np.ones_like(body)), ('body', body), ('edge', binary_dilation(body, iterations=3) & ~body), ('background', ~binary_dilation(body, iterations=3))]}
            deleted = None
            for operation in operations:
                view = n.camera_at(index, .5) if operation == 'ego_shift' else camera
                out = baseline if operation == 'noop' else n.render_operation(view, operation)
                save(dest / operation / 'native_rgb' / f'{raw:03d}.png', out['rgb'])
                Image.fromarray(union.astype('uint8') * 255).save(dest / operation / 'mask' / f'{raw:03d}.png')
                np.savez_compressed(dest / operation / 'float' / f'{raw:03d}.npz',
                                    alpha=cpu(out['alpha']), depth_sum=cpu(out['depth_sum']), dynamic_weight=cpu(out['extra']))
                if operation == 'delete_all':
                    deleted = out['rgb'].clone()
                repeat_difference = float((out['rgb'] - deleted).abs().max()) if operation == 'repeat_delete' else None
                assert repeat_difference is None or repeat_difference == 0
                rows.append(dict(raw=raw, operation=operation, generation_mask_pixels=int(union.sum()),
                                 noop_route_max_difference=identity_difference,
                                 noop_psnr_png_gt=noop_metrics if operation == 'noop' else None,
                                 repeat_max_difference=repeat_difference,
                                 outside_edit_mae=float(cpu((out['rgb'] - baseline['rgb']).abs()).mean(0)[~union].mean()) if operation != 'ego_shift' else None))
            if index % 10 == 0:
                print('SCENE_EVALUATION', name, round(time.time() - started, 2), flush=True)
    assert n.unchanged()
    summary = dict(status='complete', run=args.run, frames=1 if args.single_frame else 49,
                   operations=operations, rows=rows, new_training_updates=0,
                   master_unchanged=True, meter_to_model=n.scale,
                   fixed_module1='native GS + existing whole-dynamic edit operators',
                   fixed_module2='no new repair training; static background as deterministic baseline',
                   fixed_module3='pretrained ProPainter, same seed and parameters, invoked separately',
                   limitations=['whole modeled dynamic branch is edited; selected-object attribution remains pending',
                                'reuse insertion is not novel external asset insertion; collision not certified',
                                'all current views were reconstruction inputs; no novel-view GT claim',
                                'repeated render equality is engineering reproducibility, not persistent memory benefit',
                                'no ground-truth RGB under real occluded cars; no success inferred there'],
                   seconds=time.time() - started, cuda_peak_allocated=torch.cuda.max_memory_allocated(),
                   cuda_peak_reserved=torch.cuda.max_memory_reserved(), quality_accepted=False, human_verdict=None)
    (dest / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    from gpu_slot import exclusive
    with exclusive():
        main()
