"""Frozen PVD readout oracle; 0 updates, temporary colour extrema restored exactly."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
import argparse
import gc
import hashlib
import json
import sys
import time

RUN=Path(__file__).resolve().parent.parent


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def state_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        a=value.detach().cpu().contiguous().numpy()
        h.update(name.encode()+str(a.shape).encode()+str(a.dtype).encode()+a.tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=RUN/'formal/checkpoint001000.pt')
    p.add_argument('--output',type=Path,default=RUN/'evidence/readout_oracle')
    args=p.parse_args()
    sys.dont_write_bytecode=True
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    import c2_runtime as c2
    from baseline_audit import render_baseline
    from editor_common import raster
    started=time.monotonic();torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    ck=torch.load(str(args.checkpoint),map_location='cpu',weights_only=False)
    e=c2.Experiment(ck['contract']['settings'],Path(ck['contract']['actor_asset']))
    assert ck['parent_identity']==e.n.before and ck['step']==1000
    e.actor.load_state_dict(ck['actor'],strict=True)
    assert all(torch.equal(v.detach().cpu(),ck['actor'][k]) for k,v in e.actor.state_dict().items())
    e.actor.requires_grad_(False)
    before=state_hash(e.actor.state_dict());base_before=state_hash(e.base.repair.state_dict())
    saved_colour=e.actor.colour_logit.detach().clone()
    del ck;gc.collect()
    def rgb8(image):
        return np.uint8(np.floor(image.detach().cpu().numpy().transpose(1,2,0).clip(0,1)*255+.5))
    def alpha_summary(alpha,mask):
        values=alpha[0,mask]
        return dict(mean=float(values.mean()),min=float(values.min()),max=float(values.max()),
            median=float(values.median()),p05=float(torch.quantile(values,.05)),p95=float(torch.quantile(values,.95)),
            fraction_below01=float((values<.1).float().mean()),fraction_below05=float((values<.5).float().mean()),
            fraction_below09=float((values<.9).float().mean()))
    rows=[]
    source=Path(e.n.config['source_path'])/'image/image_0'
    with torch.no_grad():
        for raw in (105,113,130):
            out=args.output/f'{raw:03d}';out.mkdir()
            view=e.view(raw);mask=torch.as_tensor(view['visible_evidence_mask'],device='cuda',dtype=torch.bool)
            assert mask.any()
            gt_np=cv2.resize(np.asarray(Image.open(source/f'0008{raw:03d}.png').convert('RGB'))/255.,(480,320)).astype(np.float32)
            truth=torch.as_tensor(gt_np.transpose(2,0,1).copy(),device='cuda')
            camera=e.base.camera(raw)
            actor_kw=e.actor.kwargs(camera,e.matrix(raw),e.n.args,e.n.static.get_features.shape[1],e.n.args.convert_SHs_python)
            standalone=raster(camera,e.n.args,e.n.static.active_sh_degree,actor_kw,e.n.bg)
            actual=dict(full=e.render(raw,True,enabled=True),delete=e.render(raw,False,enabled=True))
            baseline=dict(full=render_baseline(e.base,raw,True,True),delete=render_baseline(e.base,raw,False,True))
            lower={};upper={}
            try:
                e.actor.colour_logit.fill_(-30)
                lower={name:e.render(raw,name=='full',enabled=True)['rgb'] for name in ('full','delete')}
                e.actor.colour_logit.fill_(30)
                upper={name:e.render(raw,name=='full',enabled=True)['rgb'] for name in ('full','delete')}
            finally:
                e.actor.colour_logit.copy_(saved_colour)
            assert torch.equal(e.actor.colour_logit,saved_colour)
            alone=standalone['alpha'];deleted=actual['delete']['actor_alpha'];full=actual['full']['actor_alpha']
            static_delete=(actual['delete']['alpha']-actual['delete']['road_alpha']-deleted).clamp_min(0)
            record=dict(raw=raw,visible_evidence_pixels=int(mask.sum()),
                visible_overlap_label_a37=int(((torch.as_tensor(view['actor_labels'],device='cuda')==5)&mask).sum()),
                actor_alpha=dict(standalone=alpha_summary(alone,mask),C2_delete=alpha_summary(deleted,mask),C2_full=alpha_summary(full,mask)),
                static_road_blocking=dict(standalone_minus_delete_mean=float((alone-deleted)[0,mask].mean()),
                    standalone_over09_joint_below05_pixels=int(((alone[0]>.9)&(deleted[0]<.5)&mask).sum()),
                    static_alpha_T_mean=float(static_delete[0,mask].mean()),road_alpha_T_mean=float(actual['delete']['road_alpha'][0,mask].mean())),
                a37_presence_effect=dict(delete_minus_full_actor_alpha_T_mean=float((deleted-full)[0,mask].mean()),
                    actor_contribution_reduced_over01_pixels=int(((deleted[0]-full[0]>.01)&mask).sum())),
                colour_range={})
            for name in ('full','delete'):
                raw_low,raw_high=lower[name],upper[name]
                assert float((raw_low-raw_high).max())<1e-5
                lo,hi=raw_low.clamp(0,1),raw_high.clamp(0,1)
                distance=(lo-truth).clamp_min(0)+(truth-hi).clamp_min(0)
                below=truth<lo;above=truth>hi
                current=actual[name]['rgb'].clamp(0,1)
                base=baseline[name]['rgb'].clamp(0,1)
                width_error=(raw_high-raw_low-actual[name]['actor_alpha'].expand(3,-1,-1)).abs()
                record['colour_range'][name]=dict(actual_C2_srgb_MAE=float((current-truth).abs()[:,mask].mean()),
                    strong_r3_srgb_MAE=float((base-truth).abs()[:,mask].mean()),
                    conditional_colour_L1_lower_bound=float(distance[:,mask].mean()),
                    fraction_channels_target_below_L=float(below[:,mask].float().mean()),
                    fraction_channels_target_above_U=float(above[:,mask].float().mean()),
                    fraction_pixels_any_channel_below_L=float(below[:,mask].any(0).float().mean()),
                    fraction_pixels_any_channel_above_U=float(above[:,mask].any(0).float().mean()),
                    mean_RGB_range_width=float((hi-lo)[:,mask].mean()),
                    affine_range_width_vs_actor_alpha_maxabs=float(width_error[:,mask].max()))
                for suffix,image in [('actual',current),('oracle_L',lo),('oracle_U',hi),('strong_r3',base)]:
                    Image.fromarray(rgb8(image)).save(out/f'{name}_{suffix}.png')
                heat=distance.mean(0,keepdim=True).expand(3,-1,-1)*5
                Image.fromarray(rgb8(heat)).save(out/f'{name}_irreducible_colour_error_x5.png')
            for name,alpha in [('standalone',alone),('delete',deleted),('full',full),('delete_static',static_delete),('delete_road',actual['delete']['road_alpha'])]:
                Image.fromarray(rgb8(alpha.expand(3,-1,-1))).save(out/f'{name}_alpha_T.png')
            Image.fromarray(np.uint8(view['visible_evidence_mask'])*255).save(out/'visible_evidence_mask.png')
            Image.fromarray(rgb8(truth)).save(out/'gt_opencv.png')
            rows.append(record)
            write(args.output/f'{raw:03d}_metrics.json',record)
            print(json.dumps(dict(stage='readout_oracle',**record)),flush=True)
    restored=before==state_hash(e.actor.state_dict())
    frozen=base_before==state_hash(e.base.repair.state_dict()) and e.unchanged()
    result=dict(optimizer_updates=0,checkpoint=str(args.checkpoint),raws=[105,113,130],
        all_actor_tensors_restored_exactly=restored,parent_and_road_unchanged=frozen,geometry_opacity_gates_fixed=True,
        oracle='Only temporary actor colour logits -30/+30; same full/delete depth-sorted renderer; original logits restored after each raw',
        interpretation='Per-pixel bounded-RGB colour fitting L1 lower bound conditional on current fixed geometry/opacity/gates. Shared multiview material constraints can make the attainable error larger.',
        not_a_method_colour_change=True,not_a_global_method_impossibility_proof=True,
        hidden_a6_GT_unavailable=True,human_verdict=None,quality_accepted=False,
        seconds=time.monotonic()-started,rows=rows)
    assert restored and frozen
    write(args.output/'summary.json',result)
    print(json.dumps(dict(stage='COMPLETE',seconds=result['seconds'],restored=restored,parent_unchanged=frozen)),flush=True)
    del e;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    main()
