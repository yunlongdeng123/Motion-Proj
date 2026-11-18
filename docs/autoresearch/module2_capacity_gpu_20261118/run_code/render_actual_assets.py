"""从公共持久资产重新原生渲染五帧；无图像补贴、无优化更新。"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from PIL import Image
from road_capacity_native import load_context,write_json,GPU_CAMPAIGN,CANONICAL_ASSET,FRAMES

def rgb8(a):return np.clip(np.rint(np.asarray(a).transpose(1,2,0)*255),0,255).astype(np.uint8)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--lease-owner',required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    import torch
    torch.set_num_threads(2)
    assets={'baseline':CANONICAL_ASSET,'I0':GPU_CAMPAIGN/'check3/I0/formal_r1/asset','W4':GPU_CAMPAIGN/'check3/W4/formal_r1/asset'}
    rows=[];begin=time.monotonic();native_calls=0
    baseline_joint={};baseline_road={}
    for name,asset in assets.items():
        ctx=load_context(asset_dir=asset,lease_file=GPU_CAMPAIGN/'gpu_lease.json',lease_owner=args.lease_owner)
        target=args.output/name;target.mkdir()
        with torch.no_grad():
            for raw in FRAMES:
                road=ctx.render_road(raw);joint=ctx.render_joint(raw)
                road_rgb=road['rgb'].cpu().numpy();alpha=road['alpha'][0].cpu().numpy();joint_rgb=joint['rgb'].cpu().numpy()
                if name=='baseline':baseline_joint[raw]=joint_rgb.copy();baseline_road[raw]=road_rgb.copy()
                K=ctx.K_masks[raw]
                difference=np.abs(joint_rgb-baseline_joint[raw]).max(0)
                if np.any(difference[K]!=0):raise RuntimeError('独立公共加载实际联合K不为0')
                Image.fromarray(rgb8(road_rgb)).save(target/f'{raw:03d}_road.webp',lossless=True)
                Image.fromarray(rgb8(joint_rgb)).save(target/f'{raw:03d}_joint.webp',lossless=True)
                np.savez_compressed(target/f'{raw:03d}_native.npz',road_rgb=road_rgb,road_alpha=alpha,joint_rgb=joint_rgb,K=K)
                expected=GPU_CAMPAIGN/f'check3/{name}/formal_r1/rendered/{raw:03d}.npz'
                error=0.
                if name!='baseline':
                    with np.load(expected,allow_pickle=False) as saved:error=float(np.abs(road_rgb-saved['rgb_srgb']).max())
                    if error>1e-6:raise RuntimeError('独立公共render与正式末步不同')
                rows.append(dict(variant=name,raw=raw,keep_pixels=int(K.sum()),joint_K_RGB_max=float(difference[K].max()),
                    road_vs_formal_float_max=error,road_vs_baseline_mean=float(np.abs(road_rgb-baseline_road[raw]).mean())))
        ctx.assert_frozen();native_calls+=ctx.native_calls;ctx.close()
    receipt=dict(schema='road_capacity_public_native_previews.v1',GPU_executed=True,optimizer_updates=0,
        public_loader=True,image_space_paste=False,frames=list(FRAMES),variants=list(assets),rows=rows,
        native_raster_calls=native_calls,seconds=time.monotonic()-begin,K_all_exact_zero=True,
        lossless_WebP=True,full_precision_native_arrays_retained=True,production_promotion=False,human_verdict=None)
    write_json(args.output/'receipt.json',receipt)
    print(json.dumps(dict(completed=True,native_calls=native_calls,seconds=receipt['seconds'],output=str(args.output))),flush=True)

if __name__=='__main__':main()
