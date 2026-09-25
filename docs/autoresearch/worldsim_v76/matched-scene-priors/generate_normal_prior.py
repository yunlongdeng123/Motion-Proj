"""按 DSINE 官方 test_minimal 路径生成 VAD-GS RGB uint8 法线先验。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

REPO = Path('/root/autodl-tmp/external/worldsim_v75/DSINE')
sys.path.insert(0, str(REPO))


def load_model(checkpoint):
    import geffnet
    from models.dsine.v02 import DSINE_v02
    # 严格加载完整 DSINE 权重会覆盖 encoder，避免另行下载冗余 ImageNet 权重。
    original = geffnet.create_model
    def without_pretrained(*args, **kwargs):
        kwargs['pretrained'] = False
        return original(*args, **kwargs)
    geffnet.create_model = without_pretrained
    config = SimpleNamespace(NNET_encoder_B=5, NNET_decoder_NF=2048,
        NNET_decoder_BN=False, NNET_decoder_down=8, NNET_learned_upsampling=True,
        NRN_prop_ps=5, NRN_num_iter_train=5, NRN_num_iter_test=5,
        NRN_ray_relu=True, NNET_output_dim=3, NNET_feature_dim=64, NNET_hidden_dim=64)
    try:
        model = DSINE_v02(config)
    finally:
        geffnet.create_model = original
    state = torch.load(checkpoint, map_location='cpu')['model']
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.cuda().eval()


def infer(model, image, intrinsics):
    from utils.utils import get_padding
    rgb = np.asarray(image).astype(np.float32) / 255
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)[None].cuda()
    h, w = rgb.shape[:2]
    padding = get_padding(h, w)
    tensor = F.pad(tensor, padding, mode='constant', value=0)
    mean = tensor.new_tensor([.485, .456, .406])[None, :, None, None]
    std = tensor.new_tensor([.229, .224, .225])[None, :, None, None]
    tensor = (tensor - mean) / std
    fx, fy, cx, cy = intrinsics[:4]
    k = tensor.new_tensor([[fx, 0, cx + padding[0]], [0, fy, cy + padding[2]], [0, 0, 1]])[None]
    normal = model(tensor, intrins=k)[-1][0, :, padding[2]:padding[2]+h, padding[0]:padding[0]+w]
    normal = normal.permute(1, 2, 0).cpu().numpy()
    if not np.isfinite(normal).all():
        raise ValueError('non-finite DSINE prediction')
    return normal


def compare_encoding(pred, reference):
    # 对比已有官方先验的轴顺序和符号；数值一致性不等于独立法线真值。
    ref = np.asarray(Image.open(reference).convert('RGB')).astype(np.float32) / 255 * 2 - 1
    assert ref.shape == pred.shape
    ref /= np.maximum(np.linalg.norm(ref, axis=-1, keepdims=True), 1e-8)
    pred = pred / np.maximum(np.linalg.norm(pred, axis=-1, keepdims=True), 1e-8)
    variants = {'identity': pred, 'negated': -pred, 'rgb_bgr_swap': pred[..., ::-1]}
    return {name: {'mean_angular_deg': float(np.degrees(np.arccos(np.clip((value*ref).sum(-1), -1, 1))).mean()),
                   'median_angular_deg': float(np.median(np.degrees(np.arccos(np.clip((value*ref).sum(-1), -1, 1)))))}
            for name, value in variants.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('/root/autodl-tmp/models/v76_dsine/dsine.pt'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--names', help='固定样例，如020_0,020_1；省略则前61帧六相机')
    parser.add_argument('--compare-official', action='store_true')
    parser.add_argument('--stop-on-training', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    output = args.output_dir or args.scene / 'normal_img'
    output.mkdir(parents=True, exist_ok=True)
    if args.compare_official and output.resolve() == (args.scene/'normal_img').resolve():
        raise ValueError('official reference must not be overwritten')
    names = args.names.split(',') if args.names else [f'{f:03d}_{c}' for f in range(61) for c in range(6)]
    report = {'scene': str(args.scene), 'seed': 0, 'checkpoint': str(args.checkpoint),
              'dsine_commit': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
              'implementation': 'DSINE_v02, official exp001 config, full native resolution, calibrated intrinsics',
              'encoding': 'RGB uint8 floor((normal+1)*127.5); VAD-GS decodes negative RGB',
              'generated': 0, 'skipped': 0, 'comparisons': {}, 'status': 'running'}
    def save():
        temporary = output / 'generation_report.tmp'
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(output / 'generation_report.json')
    model = load_model(args.checkpoint)
    start = time.monotonic()
    with torch.inference_mode():
        for name in names:
            if args.stop_on_training:
                state_path = Path('/root/autodl-tmp/runs/v76_ego_view/VADGS-P0R1-000/pipeline_state.json')
                state = json.loads(state_path.read_text())
                if state['stage'] not in ('exhaustive_matching', 'triangulation'):
                    report['status'] = 'paused_for_main_pipeline'
                    save()
                    return
            destination = output / f'{name}.png'
            if destination.exists() and not args.compare_official:
                report['skipped'] += 1
                continue
            image = Image.open(args.scene / 'images' / f'{name}.jpg').convert('RGB')
            intrinsics = np.loadtxt(args.scene / 'intrinsics' / f'{name.split("_")[1]}.txt')
            normal = infer(model, image, intrinsics)
            if args.compare_official:
                report['comparisons'][name] = compare_encoding(normal, args.scene/'normal_img'/f'{name}.png')
            encoded = np.clip((normal + 1) * 127.5, 0, 255).astype(np.uint8)
            temporary = destination.with_suffix('.tmp.png')
            Image.fromarray(encoded).save(temporary)
            temporary.replace(destination)
            report['generated'] += 1
            report['seconds'] = time.monotonic() - start
            if report['generated'] % 20 == 0:
                save()
                print(f"normal prior: {report['generated']} generated, {report['skipped']} skipped", flush=True)
    report['status'] = 'complete'
    save()
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
