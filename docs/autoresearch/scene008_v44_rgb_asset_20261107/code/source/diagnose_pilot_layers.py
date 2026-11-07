"""四帧结果的最小来源复查：三次前向补 current 的同排序 RGB 分组。

不改任何成像参数。先与已保存 RGB/alpha/depth 比较，再报告预登记银框
和目视后追加的 raw105 引擎盖开发诊断框。新框不当成正式通过标准。
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image, ImageDraw
import torch

CAM = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ART = CAM/'scene008_a6_insert_artifacts_20261105'
CONTROL = CAM/'scene008_a6_asset_controls_20261105'
sys.path[:0] = [str(CONTROL/'source'), str(ART/'source/scale')]
from scene_runtime import Runtime
from sh_asset import SHAsset
from surface_lod_asset import SurfaceLODAsset
from utils.sh_utils import eval_sh

ROIS = {105: (18, 237, 91, 262), 130: (202, 192, 298, 244), 137: (210, 213, 290, 233)}
GROUPS = ('asset', 'a37', 'road', 'native_static')


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def means(a, box):
    x0, y0, x1, y1 = box
    return a[..., y0:y1, x0:x1].mean(axis=(-2, -1), dtype=np.float64)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.time()
    rt = Runtime()
    assert rt.n.static.active_sh_degree == rt.n.dynamic.active_sh_degree == 3
    asset = SurfaceLODAsset(SHAsset.from_geometry(
        ART/'surface_fit/combined_v1/observed_fit/formal/complete_a6.npz').cuda(),
        ART/'surface_fit/masked_depth_v1/posterior.npz')
    results = []
    for raw, box in ROIS.items():
        data = rt.raw(raw)
        scene = rt.scene(raw, asset, delete=True)
        rows = scene['rows']
        if rows['colors_precomp'] is None:
            direction = rows['means3D']-data['cam'].camera_center
            direction = direction/direction.norm(dim=-1, keepdim=True)
            colour = (eval_sh(3, rows['shs'].transpose(1, 2), direction)+.5).clamp_min(0)
        else:
            colour = rows['colors_precomp']
        feature = rows['features'].clone()
        for index in range(4):
            feature[:, 6+3*index:9+3*index] = colour*feature[:, 2+index:3+index]
        scene['rows'] = dict(rows, features=feature)
        out = rt.render_rows(raw, scene)
        rgb = out['rgb'].cpu().numpy()
        alpha = out['alpha'][0].cpu().numpy()
        depth = (out['depth_sum'][0]/out['alpha'][0].clamp_min(1e-8)/rt.scale).cpu().numpy()
        with np.load(args.results/f'current/{raw:03d}/delete_float.npz') as saved:
            parity = dict(RGB=float(abs(rgb-saved['rgb']).max()),
                          alpha=float(abs(alpha-saved['alpha']).max()),
                          depth_m=float(abs(depth-saved['depth_m']).max()))
        assert max(parity.values()) < 2e-5, parity
        current = dict(group_rgb_T=out['extra'][4:16].reshape(4, 3, 320, 480).cpu().numpy(),
                       group_alpha_T=out['component_alpha_T'].cpu().numpy(),
                       env_rgb_T=((1-out['alpha'])*data['env']).cpu().numpy(),
                       constant_background_rgb_T=((1-out['alpha'])*rt.n.bg[:, None, None]).cpu().numpy())
        reconstruction = current['group_rgb_T'].sum(0)+current['env_rgb_T']+current['constant_background_rgb_T']
        error = float(abs(reconstruction-rgb).max())
        assert error < 5e-5
        with np.load(args.results/f'v44/{raw:03d}/delete_components.npz') as saved:
            candidate = {k: saved[k].copy() for k in saved.files}
        with np.load(args.results/f'v44/{raw:03d}/delete_float.npz') as saved:
            new_rgb = saved['rgb'].copy()
        row = dict(raw=raw, xyxy=box, pixels=(box[2]-box[0])*(box[3]-box[1]),
                   role='posthoc_hood_development_diagnostic' if raw == 105 else 'predeclared_whole_silver_ROI',
                   marker_parity_maxabs=parity, current_reconstruction_maxabs=error, groups={})
        for index, name in enumerate(GROUPS):
            b = means(current['group_rgb_T'][index], box)
            a = means(candidate['group_rgb_T'][index], box)
            row['groups'][name] = dict(current_RGB=b.tolist(), v44_RGB=a.tolist(),
                delta_RGB=(a-b).tolist(), delta_mean_brightness=float((a-b).mean()),
                current_alpha_T=float(means(current['group_alpha_T'][index], box)),
                v44_alpha_T=float(means(candidate['group_alpha_T'][index], box)))
        for name in ('env_rgb_T', 'constant_background_rgb_T'):
            b, a = means(current[name], box), means(candidate[name], box)
            row['groups'][name] = dict(current_RGB=b.tolist(), v44_RGB=a.tolist(),
                delta_RGB=(a-b).tolist(), delta_mean_brightness=float((a-b).mean()))
        net = means(new_rgb-rgb, box)
        sum_delta = np.sum([np.array(x['delta_RGB']) for x in row['groups'].values()], axis=0)
        row['output_delta_RGB'] = net.tolist()
        row['sum_delta_RGB'] = sum_delta.tolist()
        row['delta_additivity_maxabs'] = float(abs(net-sum_delta).max())
        assert row['delta_additivity_maxabs'] < 5e-5
        # 同曝光展示，不对单层做自动提亮；图像只作同排序颜色组成可视化。
        x0, y0, x1, y1 = box
        arrays = [rgb, new_rgb, current['group_rgb_T'][3], candidate['group_rgb_T'][3],
                  current['env_rgb_T'], candidate['env_rgb_T']]
        labels = ['current output', 'v44 output', 'current static*T', 'v44 static*T',
                  'current env*T', 'v44 env*T']
        width = 240
        height = round((y1-y0)/(x1-x0)*width)
        board = Image.new('RGB', (width*3, (height+30)*2), '#f0f4f8')
        draw = ImageDraw.Draw(board)
        for i, (array, label) in enumerate(zip(arrays, labels)):
            im = Image.fromarray(np.uint8(np.clip(array[:, y0:y1, x0:x1].transpose(1, 2, 0), 0, 1)*255+.5))
            xx, yy = i%3*width, i//3*(height+30)
            draw.text((xx+6, yy+7), label, fill='black')
            board.paste(im.resize((width, height), Image.Resampling.NEAREST), (xx, yy+30))
        board.save(args.output/f'raw{raw}_same_sort_layers.png')
        results.append(row)
        rt._data.clear()
        torch.cuda.empty_cache()
    assert rt.modelunchanged() and asset.assert_persistent_unchanged()
    receipt = dict(status='PASS_THREE_FORWARD_SOURCE_DIAGNOSTICS', rows=results,
                   actual_native_forward_calls=3, training_updates=0, method_parameters_changed=False,
                   source_colours_or_opacities_or_geometry_updated=False, shutdown_performed=False,
                   seconds=time.time()-start, intrinsic_environment_vs_transmittance_not_separated=True,
                   cause_of_training_artifacts_not_proven=True)
    write(args.output/'receipt.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
