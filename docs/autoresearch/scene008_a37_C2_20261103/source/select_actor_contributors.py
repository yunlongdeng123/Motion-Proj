"""Zero-update static alpha*T contributor candidates, never ownership labels."""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
from pathlib import Path
import argparse
import gc
import hashlib
import json
import sys
import time

RUN = Path(__file__).resolve().parent.parent
R3 = RUN.parent / 'scene008_a37_only_B10k_road_20261103_r3'


def write(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset',type=Path,default=RUN/'inputs/actor_surface_point_seed0_diagnostic.npz')
    parser.add_argument('--output',type=Path,default=RUN/'inputs/actor_contributors_point.npz')
    parser.add_argument('--evidence',type=Path,default=RUN/'evidence/actor_contributors_point')
    args=parser.parse_args()
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(R3/'source'))
    import numpy as np
    from PIL import Image
    import torch
    import repair37
    import c2_runtime
    from baseline_audit import render_baseline
    from editor_common import prepare_rasterization,road_kwargs,merge_rows,raster
    assert not args.output.exists(), 'Preserve an existing contributor selection; use a new explicit output path'
    args.evidence.mkdir(parents=True,exist_ok=False)
    (args.evidence/'masks').mkdir()
    started=time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    base=repair37.Experiment()
    ck=torch.load(str(R3/'formal/checkpoint001000.pt'),map_location='cpu',weights_only=False)
    assert ck['step']==1000 and ck['parent_identity']==base.n.before
    base.repair.load_state_dict(ck['repair'],strict=True)
    base.repair.requires_grad_(False)
    del ck
    actor=c2_runtime.ActorSurface(args.asset)
    actor.requires_grad_(False)
    actor.eval()
    geometry=dict(np.load(RUN/'inputs/a6_geometry_minimal.npz',allow_pickle=False))
    n=base.n
    count=len(n.static.get_xyz)
    assert count==1520544, f'Unexpected static row binding: {count}'
    identity=dict(n.before)
    fit=torch.zeros(count,device='cuda')
    guard=torch.zeros_like(fit)
    fit_full=torch.zeros_like(fit)
    fit_delete=torch.zeros_like(fit)
    guard_full=torch.zeros_like(fit)
    guard_delete=torch.zeros_like(fit)
    sum_fit=torch.zeros(count,device='cuda',dtype=torch.float64)
    sum_guard=torch.zeros_like(sum_fit)
    rows=[]
    rgb_max=0.0
    asset_receipt=args.asset.with_name(args.asset.stem+'_receipt.json')
    receipt=json.loads(asset_receipt.read_text()) if asset_receipt.exists() else {}
    contract=dict(optimizer_updates=0,static_rows=count,parent_identity=identity,
        frozen_parent='single-a37 B10k static plus frozen r3 gate/road and current-time a37 dynamic',
        asset=str(args.asset),asset_formal_eligible=receipt.get('formal_eligible'),
        source_asset_sha256=hashlib.sha256(args.asset.read_bytes()).hexdigest(),
        observed_mask='a6_views actor_labels==4, all49 observed visible-label construction',
        amodal_mask='standalone initial ActorSurface proposal raster alpha>0.10, without other scene objects; proposal-dependent domain, not truth',
        full_guard='labels1..3 OR a37body(label5) OR outside standalone actor amodal mask',
        delete_guard='labels1..3 OR outside standalone actor amodal mask',
        score='Exact static alpha*T by zero-valued per-row feature gradient; sum full/delete within each frame, maximum over49 frames',
        candidate='fit_score>1e-4 only; guard conflict never rejects or silently narrows candidates',
        candidate_is_not_ownership_GT=True,candidates_are_not_deleted=True,raw_range=[89,137],
        geometry_and_native_time_unchanged=True,all_inputs_BUILD=True,independent_test=False)
    write(args.evidence/'contract.json',contract)

    def score(raw,include_target,positive,protected):
        camera=base.camera(raw)
        kw=prepare_rasterization(camera,n.static,n.args)
        kw['means2D']=torch.zeros_like(kw['means3D'])
        ids=base.repair.ids
        kw['opacities']=kw['opacities'].index_copy(0,ids,kw['opacities'][ids]*base.repair.gate_raw.sigmoid()[:,None])
        marker=torch.zeros((count,2),device='cuda',requires_grad=True)
        kw['features']=torch.cat((kw['features'],marker),1)
        if include_target:
            dynamic=prepare_rasterization(camera,n.dynamic,n.args,dynamic=True)
            dynamic['means2D']=torch.zeros_like(dynamic['means3D'])
            dynamic['features']=torch.cat((dynamic['features'],torch.zeros((len(dynamic['features']),2),device='cuda')),1)
            kw=merge_rows(kw,dynamic)
        channels=kw['shs'].shape[1] if kw['shs'] is not None else 1
        road=road_kwargs(camera,base.repair.road,n.args,channels,precomputed_color=kw['colors_precomp'] is not None)
        road['features']=torch.cat((road['features'],torch.zeros((len(road['features']),2),device='cuda')),1)
        out=raster(camera,n.args,n.static.active_sh_degree,merge_rows(kw,road),n.bg)
        env=n.env(camera.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
        rgb=out['rgb']+(1-out['alpha'])*env
        with torch.no_grad():
            reference=render_baseline(base,raw,include_target,True)['rgb']
            difference=float((rgb-reference).abs().max())
            assert difference==0.0, (raw,include_target,'marker altered RGB',difference)
        positive=torch.as_tensor(positive,device='cuda',dtype=torch.bool)
        protected=torch.as_tensor(protected,device='cuda',dtype=torch.bool)
        value=out['extra'][0,positive].sum()+out['extra'][1,protected].sum()
        gradient=torch.autograd.grad(value,marker,only_inputs=True)[0].detach()
        assert torch.isfinite(gradient).all()
        assert float(gradient.min())>=0, 'alpha*T contributor weights must be nonnegative'
        result=(gradient[:,0].clone(),gradient[:,1].clone(),difference)
        del kw,road,out,marker,rgb,reference,gradient
        return result

    with (args.evidence/'frames.jsonl').open('x') as log:
        for raw in range(89,138):
            labels=np.load(RUN/'inputs/a6_views'/f'{raw:03d}.npz',allow_pickle=False)['actor_labels']
            observed=labels==4
            camera=base.camera(raw)
            with torch.no_grad():
                channels=n.static.get_features.shape[1]
                actor_kwargs=actor.kwargs(camera,torch.as_tensor(geometry['model_from_canonical'][raw],device='cuda',dtype=torch.float32),n.args,channels,n.args.convert_SHs_python)
                standalone=raster(camera,n.args,n.static.active_sh_degree,actor_kwargs,n.bg)
                amodal=(standalone['alpha'][0]>.10).cpu().numpy()
            other=(labels>=1)&(labels<=3)
            full_protected=other|(labels==5)|~amodal
            delete_protected=other|~amodal
            full_fit,full_guard,full_diff=score(raw,True,observed,full_protected)
            delete_fit,delete_guard,delete_diff=score(raw,False,amodal,delete_protected)
            combined_fit=full_fit+delete_fit
            combined_guard=full_guard+delete_guard
            fit=torch.maximum(fit,combined_fit)
            guard=torch.maximum(guard,combined_guard)
            fit_full=torch.maximum(fit_full,full_fit)
            fit_delete=torch.maximum(fit_delete,delete_fit)
            guard_full=torch.maximum(guard_full,full_guard)
            guard_delete=torch.maximum(guard_delete,delete_guard)
            sum_fit+=combined_fit.double()
            sum_guard+=combined_guard.double()
            rgb_max=max(rgb_max,full_diff,delete_diff)
            record=dict(raw=raw,observed_pixels=int(observed.sum()),amodal_pixels=int(amodal.sum()),
                amodal_unobserved_pixels=int((amodal&~observed).sum()),observed_outside_amodal=int((observed&~amodal).sum()),
                full_guard_pixels=int(full_protected.sum()),delete_guard_pixels=int(delete_protected.sum()),
                observed_guard_overlap=int((observed&full_protected).sum()),
                amodal_guard_overlap=int((amodal&delete_protected).sum()),
                full_fit_alpha_T_sum=float(full_fit.sum()),delete_fit_alpha_T_sum=float(delete_fit.sum()),
                full_guard_alpha_T_sum=float(full_guard.sum()),delete_guard_alpha_T_sum=float(delete_guard.sum()),
                frame_fit_rows_over1e4=int((combined_fit>1e-4).sum()),rgb_marker_maxabs=max(full_diff,delete_diff),
                seconds=time.monotonic()-started)
            rows.append(record)
            log.write(json.dumps(record)+'\n');log.flush()
            for key,mask in [('observed',observed),('amodal',amodal),('guard_full',full_protected),('guard_delete',delete_protected)]:
                Image.fromarray(np.uint8(mask)*255).save(args.evidence/'masks'/f'{raw:03d}_{key}.png')
            if raw==89 or (raw-88)%10==0 or raw==137:
                print(json.dumps(dict(stage='contributor_selection',**record)),flush=True)
            del standalone,actor_kwargs,full_fit,full_guard,delete_fit,delete_guard,combined_fit,combined_guard
    selected=fit>1e-4
    ids=np.flatnonzero(selected.cpu().numpy())
    assert len(ids)>0
    tensor_cpu=lambda value:value.detach().cpu().numpy()
    selected_fit=fit[selected]
    selected_guard=guard[selected]
    ratio=selected_guard/selected_fit
    conflicts={str(multiplier):dict(rows=int((selected_guard>multiplier*selected_fit).sum()),
        fraction=float((selected_guard>multiplier*selected_fit).float().mean())) for multiplier in (0,.1,1,20)}
    summary=dict(contract=contract,frame_count=49,candidate_rows=len(ids),candidate_fraction=len(ids)/count,
        guard_fit_conflicts=conflicts,guard_fit_ratio_quantiles={str(q):float(torch.quantile(ratio,q)) for q in (0.,.5,.9,.99,1.)},
        fit_alpha_T_flow_total=float(sum_fit.sum()),fit_alpha_T_flow_on_candidates=float(sum_fit[selected].sum()),
        fit_flow_retained_fraction=float(sum_fit[selected].sum()/sum_fit.sum()),
        guard_alpha_T_flow_total=float(sum_guard.sum()),guard_alpha_T_flow_on_candidates=float(sum_guard[selected].sum()),
        guard_flow_candidate_fraction=float(sum_guard[selected].sum()/sum_guard.sum()),
        parent_unchanged=base.unchanged(),rgb_marker_maxabs=rgb_max,optimizer_updates=0,
        output=str(args.output),seconds=time.monotonic()-started,
        candidate_is_not_ownership_GT=True,candidates_are_not_deleted=True)
    assert summary['parent_unchanged'] and rgb_max==0
    temporary=args.output.with_name(args.output.name+'.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,ids=ids,fit_score=tensor_cpu(selected_fit),guard_score=tensor_cpu(selected_guard),
            full_visible_score=tensor_cpu(fit_full[selected]),delete_amodal_score=tensor_cpu(fit_delete[selected]),
            full_guard_score=tensor_cpu(guard_full[selected]),delete_guard_score=tensor_cpu(guard_delete[selected]),
            fit_sum=tensor_cpu(sum_fit[selected]),guard_sum=tensor_cpu(sum_guard[selected]),static_rows=np.int64(count),
            parent_identity_json=np.asarray(json.dumps(identity,sort_keys=True)),asset=np.asarray(str(args.asset)))
    os.replace(temporary,args.output)
    summary['output_sha256']=hashlib.sha256(args.output.read_bytes()).hexdigest()
    write(args.evidence/'summary.json',summary)
    write(args.evidence/'metrics.json',dict(summary=summary,frames=rows))
    write(args.output.with_name(args.output.stem+'_receipt.json'),summary)
    print(json.dumps(dict(stage='COMPLETE',summary=summary)),flush=True)
    del actor,base
    gc.collect()
    torch.cuda.empty_cache()


if __name__=='__main__':
    main()
