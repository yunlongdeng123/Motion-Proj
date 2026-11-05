"""Read-only a5 stage ablation. No SGD, no model tensor modifications."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ASSET_SOURCE = CAMPAIGN/'scene008_a6_asset_controls_20261105/source'
OLD = CAMPAIGN/'scene008_a37_only_B10k_20261103_r1/formal10k'
OBS = CAMPAIGN/'scene008_instances_20261105_r1/observations/frames'
RAWS = (89, 97, 105, 109)
BOX = (0, 160, 175, 320)  # View only; metric mask is actual a5 label1.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('oldclean', 'current'), required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(1)
    stages = []
    if args.mode == 'oldclean':
        sys.path.insert(0, str(CAMPAIGN/'scene008_native_road1k_20261105_r1/source'))
        import editor_common as common
        assert common.B == OLD
        n = common.Native()
        versions = {(name,key): (id(value), value._version) for name,model in (('static',n.static), ('dynamic',n.dynamic), ('env',n.env))
                    for key, value in vars(model).items() if torch.is_tensor(value)}
        runtime = None
        asset = None
    else:
        sys.path.insert(0, str(ASSET_SOURCE))
        from scene_runtime import Runtime
        from sh_asset import SHAsset
        runtime = Runtime(fine_road=True)
        n = runtime.n
        asset = SHAsset.from_geometry(ASSET_SOURCE.parent/'m2/complete_a6.npz').cuda().requires_grad_(False)
        asset_before = {key: value.detach().cpu().clone() for key, value in
                        (*asset.named_parameters(), *asset.named_buffers())}
        assert len(asset_before) == 27
    x0,y0,x1,y1 = BOX
    for raw in RAWS:
        with np.load(OBS/f'{raw:03d}.npz', allow_pickle=False) as data:
            obs = {key: data[key].copy() for key in data.files}
        cam = n.camera(raw)
        assert abs(float(obs['model_time'])-cam.timestamp) < 1e-6
        np.testing.assert_allclose(obs['model_to_camera'].T, cam.world_view_transform.cpu().numpy(), atol=1e-6, rtol=0)
        gt = obs['rgb'].astype(np.float32).transpose(2,0,1)
        if gt.max() > 1: gt /= 255.
        mask = obs['actor_labels'] == 1
        Image.fromarray(np.uint8(gt.transpose(1,2,0)*255+.5)).save(args.output/f'raw{raw}_gt.png')
        outputs = {}

        def save(stage, out):
            assert bool(torch.isfinite(out['rgb']).all())
            rgb = out['rgb'].detach().cpu().numpy()
            alpha = out['alpha'].detach().cpu().numpy()
            Image.fromarray(np.uint8(np.clip(rgb.transpose(1,2,0),0,1)*255+.5)).save(args.output/f'raw{raw}_{stage}.png')
            error = rgb-gt
            record = dict(raw=raw, stage=stage, a5_label1_pixels=int(mask.sum()),
                          a5_RGB_MAE=float(np.abs(error[:,mask]).mean()) if mask.any() else None,
                          a5_RGB_mean=rgb[:,mask].mean(1).tolist() if mask.any() else None,
                          a5_alpha_mean=float(alpha[:,mask].mean()) if mask.any() else None,
                          a5_alpha_lt_05=int((alpha[:,mask]<.5).sum()),
                          timestamp=float(cam.timestamp))
            if outputs:
                previous = next(reversed(outputs))
                delta = rgb-outputs[previous]
                record['previous_stage'] = previous
                record['a5_delta_maxabs'] = float(np.abs(delta[:,mask]).max()) if mask.any() else None
                record['a5_delta_MAE'] = float(np.abs(delta[:,mask]).mean()) if mask.any() else None
            np.savez_compressed(args.output/f'raw{raw}_{stage}_a5.npz',
                xyxy=BOX, rgb=rgb[:,y0:y1,x0:x1], target=gt[:,y0:y1,x0:x1],
                alpha=alpha[:,y0:y1,x0:x1], a5_mask=mask[y0:y1,x0:x1],
                labels=obs['actor_labels'][y0:y1,x0:x1])
            outputs[stage] = rgb
            stages.append(record)
            print(f'RENDER {args.mode} raw{raw} {stage} a5MAE={record["a5_RGB_MAE"]}', flush=True)

        with torch.no_grad():
            save('native_keep', n.render(cam, dynamic=True))
            save('native_delete', n.render(cam, dynamic=False))
            if runtime is not None:
                scene = runtime.scene(raw, baseline=True, delete=True)
                rows = scene['rows']
                road_mask = rows['features'][:,4].bool()
                assert road_mask.sum() == len(runtime.road_colors)
                no_road = dict(rows, mask=rows['mask'] & ~road_mask)
                save('gate_only_delete', runtime.render_rows(raw, dict(scene, rows=no_road)))
                coarse_rows = dict(rows)
                if rows['shs'] is not None:
                    sh = rows['shs'].clone()
                    sh[road_mask,0] = (runtime.exp.repair.road.colors()-.5)/.28209479177387814
                    coarse_rows['shs'] = sh
                else:
                    colors = rows['colors_precomp'].clone()
                    colors[road_mask] = runtime.exp.repair.road.colors()
                    coarse_rows['colors_precomp'] = colors
                save('road_coarse_delete', runtime.render_rows(raw, dict(scene, rows=coarse_rows)))
                save('road_fine_delete', runtime.render_rows(raw, scene))
                save('M2_delete', runtime.render_scene(raw, asset, delete=True))
                save('M2_keep', runtime.render_scene(raw, asset, delete=False))
                runtime.assert_unchanged()
    assert n.unchanged()
    if runtime is not None:
        assert runtime.modelunchanged()
        current = dict((*asset.named_parameters(), *asset.named_buffers()))
        for key, before in asset_before.items():
            assert torch.equal(before, current[key].detach().cpu()), key
    else:
        assert versions == {(name,key): (id(value), value._version) for name,model in (('static',n.static), ('dynamic',n.dynamic), ('env',n.env))
                            for key,value in vars(model).items() if torch.is_tensor(value)}
    record = dict(status='PASS_READ_ONLY_STAGE_ABLATION', mode=args.mode, raw_frames=RAWS,
                  training_updates=0, model_files_unchanged=True, model_parameters_unchanged=True,
                  M2_all27_unchanged=runtime is not None, a5_mask='Original observation actor_labels==1',
                  view_crop_only=BOX, rows=stages, parent=n.before,
                  script_sha256=sha(__file__))
    (args.output/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__ == '__main__':
    main()
