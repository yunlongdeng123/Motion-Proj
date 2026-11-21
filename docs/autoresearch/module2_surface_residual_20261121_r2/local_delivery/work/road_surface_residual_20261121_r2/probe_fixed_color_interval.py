"""Four fixed-geometry road-only native forwards; no formal state mutation."""
from pathlib import Path
import hashlib,json,sys,time,traceback
import numpy as np
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ROOT=C/'road_surface_residual_20261121_r2';R1=C/'road_surface_adaptive_20261121_r1'
ASSET=ROOT/'appearance_fit/asset';OUT=ROOT/'diagnostic/fixed_color_interval'
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
sys.path[:0]=[str(R1/'code'),str(REPO/'scripts/worldsim_v82'),str(C/'road_capacity_gpu_20261118_r1/code')]
from run_adaptive import geometry_render
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');q.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'receipt.json').exists():raise FileExistsError('preserve completed diagnostic; do not spend four additional nativeF')
    watched=[ASSET/'road.npz',ASSET/'provenance.npz',ASSET/'persistent/scene.json',ASSET/'persistent/proof.json',
        ROOT/'resource_ledger.json',ROOT/'status.json',ROOT/'contract.json',ROOT/'export_manifest.json',ROOT/'receipt.json',
        ROOT/'train_inputs/allowed_color_world_rows.npz']
    before={str(p):sha(p) for p in watched};scope=load(watched[-1]);ids=scope['world_row_ids'].astype(np.int64)
    assert len(ids)==1191 and len(np.unique(ids))==1191
    assert dict(zip(*np.unique(scope['group'],return_counts=True)))=={1:13,2:1071,3:107}
    lease=OUT/'gpu_lease.json';write(lease,dict(active=True,owner='r2_fixed_color_interval',gpu_authorized=True,optimizer_updates_allowed=0))
    start=time.monotonic();ctx=None;rows=[]
    try:
        import torch
        torch.set_num_threads(2)
        from drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff
        from road_capacity_native import NativeRoadContext
        scene,manifest,loader=load_persistent_handoff(ASSET/'persistent/scene.json',ASSET/'persistent/proof.json')
        assert manifest.data['render_wh']==[1152,768]
        for obj in (scene.static,scene.dynamic,scene.environment):
            for p in obj.parameters() if hasattr(obj,'parameters') else ():p.requires_grad_(False)
        ctx=NativeRoadContext(scene,manifest,loader,ASSET,(86,110),lease,'r2_fixed_color_interval')
        positions=np.searchsorted(ctx.active_world_rows,ids);assert np.array_equal(ctx.active_world_rows[positions],ids)
        pos=torch.as_tensor(positions,device='cuda',dtype=torch.long)
        base=ctx.base_rgb.detach();black=base.index_copy(0,pos,torch.zeros((len(pos),3),device='cuda'))
        white=base.index_copy(0,pos,torch.ones((len(pos),3),device='cuda'))
        untouched=np.ones(len(base),bool);untouched[positions]=False
        assert torch.equal(black[untouched],base[untouched]) and torch.equal(white[untouched],base[untouched])
        with torch.no_grad():
            for raw in (86,110):
                zero=geometry_render(ctx,raw,colors=black);one=geometry_render(ctx,raw,colors=white)
                b=zero['rgb'].cpu().numpy();r1=one['rgb'].cpu().numpy();alpha=zero['alpha'].cpu().numpy();alpha_one=one['alpha'].cpu().numpy();weight=r1-b
                assert b.shape==(3,768,1152) and alpha.shape==(768,1152) and b.dtype==np.float32
                assert np.array_equal(alpha,alpha_one),'colors unexpectedly changed the native alpha or ordering'
                formal=load(ROOT/'appearance_fit/rendered'/f'{raw:03d}.npz')
                alpha_error=float(np.abs(alpha-formal['road_alpha']).max());assert alpha_error==0
                upper=b+weight;rgb=formal['road_rgb'];below=np.maximum(b-rgb,0);above=np.maximum(rgb-upper,0)
                row=dict(raw=raw,nativeF=2,alpha_exact_formal=True,alpha_formal_max_error=alpha_error,
                    allowed_weight_channel_span_max=float((weight.max(0)-weight.min(0)).max()),
                    allowed_weight_min=float(weight.min()),allowed_weight_max=float(weight.max()),
                    allowed_weight_excess_alpha_max=float(np.maximum(weight-alpha[None],0).max()),
                    formal_RGB_interval_violation_max=float(max(below.max(),above.max())),
                    interpretation='per-pixel componentwise interval with fixed final geometry/opacity/scales/sort; not global capacity or HF feasibility')
                assert row['formal_RGB_interval_violation_max']<5e-6
                np.savez_compressed(OUT/f'{raw:03d}.npz',frozen_RGB=b,allowed_weight=weight,alpha=alpha,
                    allowed_world_row_ids=ids,allowed_group=scope['group'])
                rows.append(row)
        assert ctx.native_calls==4
        after={str(p):sha(p) for p in watched};assert before==after,'formal asset or ledger changed'
        receipt=dict(schema='fixed_color_interval_native.v1',complete=True,source_asset=str(ASSET),resolution=[1152,768],
            nativeF=4,nativeB=0,optimizer_updates=0,allowed_rows=1191,activated_existing=13,appended_new_geometry=1071,reuse_existing_newgroup=107,
            renderer='actual_ordinary_GS_road_only_S2_no_feature_override',same_geometry_alpha_sort=True,
            source_SHA256=before,formal_sources_unchanged=True,script_SHA256=sha(__file__),
            allowed_ID_array_SHA256=hashlib.sha256(ids.tobytes()).hexdigest(),rows=rows,
            output_SHA256={str(r):sha(OUT/f'{r:03d}.npz') for r in (86,110)},seconds=time.monotonic()-start,
            interval_definition='b=render(allowed RGB0); w=render(allowed RGB1)-b, retained as three FP32 channels without averaging',
            limitations='Componentwise per-pixel interval only; not simultaneous multi-view fit, global capacity or HF proof',GPU_keep_on=True)
        write(OUT/'receipt.json',receipt);print(json.dumps(receipt),flush=True)
    except Exception:
        write(OUT/'failure.json',dict(error=traceback.format_exc(),nativeF=0 if ctx is None else ctx.native_calls,nativeB=0,optimizer_updates=0));raise
    finally:
        if ctx is not None:ctx.close()
        write(lease,dict(active=False,owner='r2_fixed_color_interval',gpu_authorized=True,optimizer_updates_allowed=0))
if __name__=='__main__':main()
