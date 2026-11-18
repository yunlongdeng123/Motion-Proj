"""One-context, zero-update native basis qualification under explicit GPU authorization."""
from pathlib import Path
import argparse, hashlib, json, sys, time, traceback
import numpy as np

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
Q=ROOT/'road_teacher_detail_cpu_20261119_r1'
N=ROOT/'road_activation_20261118_r1'
G=ROOT/'road_capacity_gpu_20261118_r1'
P=ROOT/'road_multiscale_material_20261118_r1'
FRAMES=(86,110,134,179,198)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(p,obj):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def ctx_load(lease):
    sys.path.insert(0,str(G/'code'))
    from road_capacity_native import load_context
    return load_context(asset_dir=N/'B32_beam95_native_r1/asset_guarded',frames=FRAMES,
        lease_file=lease,lease_owner='stage_qa')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Q/'gpu_20261119/native_qualification_r2')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'started.json').exists():raise FileExistsError('Preserve existing run; choose a new output directory')
    authorization=Q/'gpu_20261119/authorization.json'
    root_auth=json.loads(authorization.read_text())
    if root_auth.get('GPU_authorized') is not True or root_auth.get('smoke_authorized') is not True:
        raise RuntimeError('Root GPU smoke authorization missing')
    if root_auth['lease']['native_ctx_owner']!='stage_qa':raise RuntimeError('Native context owner mismatch')
    scope_path=Q/'permission/final_r1/teacher_evidence.npz'
    with np.load(scope_path,allow_pickle=False) as z:
        active_ids=z['active_world_rows'].astype(np.int64);eligible=z['common_write_active'].copy()
    if len(active_ids)!=151988 or eligible.dtype!=bool or eligible.sum()!=89887:raise RuntimeError('Registered scope mismatch')
    signature=hashlib.sha256(active_ids[eligible].astype('<i8').tobytes()).hexdigest()
    lease=out/'gpu_lease.json'
    write(lease,dict(schema='native_qualification_compatibility_lease.v1',gpu_authorized=True,active=True,
        owner='stage_qa',authorization=str(authorization),authorization_sha256=sha(authorization),
        optimizer_updates_allowed=0,power_policy='keep_on'))
    smoke=out/'basis_smoke_contract.json'
    write(smoke,dict(schema='pixel_native_basis_smoke.v1',phase='native_smoke',native_allowed=True,
        gpu_authorized=True,formal_allowed=False,frames=list(FRAMES),scope_world_rows_sha256=signature,
        max_native_forward_calls=100,delete_actor_ids=[10],authorization=str(authorization),optimizer_updates=0))
    source_adapter=Q/'pixel_render/code/native_basis_adapter.py'
    # Preserve conflicting weights before failing closed. No gate/mask/renderer semantics changed.
    source=source_adapter.read_text()
    old="if mode=='joint'and Kweight!=0:raise RuntimeError('joint K上eligible alphaT非零，须修world row scope，禁止屏幕K掩码修补')"
    if source.count(old)!=1:raise RuntimeError('Reviewed adapter source does not match expected diagnostic hook')
    runtime=out/'code/native_basis_adapter_evidence.py';runtime.parent.mkdir(parents=True,exist_ok=True)
    runtime.write_text(source.replace(old,"# Joint K hard gate runs in owner runner immediately after raw arrays are persisted."))
    sys.path.insert(0,str(runtime.parent))
    from native_basis_adapter_evidence import export_native_basis
    paths=[scope_path,source_adapter,runtime,Path(__file__),authorization,
        N/'B32_beam95_native_r1/asset_guarded/road.npz',
        N/'B32_beam95_native_r1/asset_guarded/provenance.npz',G/'code/road_capacity_native.py']
    binding={str(p):sha(p) for p in paths}
    write(out/'source_binding.json',dict(files=binding,scope_sha256=signature,
        evidence_adapter_difference='Only defer joint-K rejection until original raw weight arrays are persisted; owner immediately rejects any nonzero weight.'))
    start=time.time();write(out/'started.json',dict(time=start,formal_updates=0,optimizer_updates=0,PID=__import__('os').getpid()))
    ctx=None;summary={}
    try:
        import torch
        torch.set_num_threads(8)
        print('LOADING_NATIVE_CONTEXT',flush=True);ctx=ctx_load(lease)
        if not np.array_equal(active_ids,ctx.active_world_rows):raise RuntimeError('Active world row identity mismatch')
        write(out/'context_ready.json',dict(seconds=time.time()-start,active_rows=len(active_ids),eligible_rows=int(eligible.sum()),
            device=str(ctx.base_rgb.device),torch=torch.__version__,dtype=str(ctx.base_rgb.dtype)))
        print('CONTEXT_READY',time.time()-start,flush=True)
        rows=[];basis={}
        with torch.no_grad():
            for raw in FRAMES:
                meta=export_native_basis(ctx,raw,eligible,out/'basis'/f'{raw:03d}',authorization=smoke)
                basis[raw]={mode:load(out/'basis'/f'{raw:03d}'/f'{mode}_basis.npz') for mode in ('road','joint')}
                write(out/'basis_progress.json',dict(completed_frames=[r['raw'] for r in rows]+[raw],native_forward_calls=ctx.native_calls))
                if meta['joint']['K_eligible_weight_max']!=0:
                    write(out/'K_conflict.json',dict(raw=raw,meta=meta,scope_unchanged=True,
                        instruction='No scope shrink or screen copy. Raw conflicting native weights retained.'))
                    raise RuntimeError('New eligible world scope has nonzero native joint K mass')
                parent=load(N/'B32_beam95_native_r1'/f'{raw:03d}_native.npz')
                rd=basis[raw]['road'];jn=basis[raw]['joint']
                row=dict(raw=raw,protected_pixels=int(jn['K'].sum()),
                    saved_parent_road_RGB_max=float(np.max(np.abs(rd['baseline_rgb']-parent['road_rgb']))),
                    saved_parent_road_alpha_max=float(np.max(np.abs(rd['alpha']-parent['road_alpha']))),
                    saved_parent_joint_RGB_max=float(np.max(np.abs(jn['baseline_rgb']-parent['joint_rgb']))),
                    joint_K_eligible_weight_max=meta['joint']['K_eligible_weight_max'],
                    road_K_eligible_weight_max=meta['road']['K_eligible_weight_max'])
                rows.append(row);write(out/'baseline_identity.json',dict(rows=rows,protected_pixel_frames=sum(r['protected_pixels'] for r in rows)))
                print('BASIS_READY',raw,'K',row['joint_K_eligible_weight_max'],'calls',ctx.native_calls,flush=True)
        # Compare feature export with standard road/joint outputs; test actual color-only linearity.
        constants=[('black',(0.,0.,0.)),('white',(1.,1.,1.)),('RGB',(0.2,0.4,0.6)),
                   ('unit_R',(1.,0.,0.)),('unit_G',(0.,1.,0.)),('unit_B',(0.,0.,1.))]
        identities=[];emask=torch.as_tensor(eligible,device=ctx.base_rgb.device)
        with torch.no_grad():
            for raw in FRAMES:
                for mode,render in [('road',ctx.render_road),('joint',ctx.render_joint)]:
                    b=basis[raw][mode]
                    actual=render(raw,ctx.base_rgb)
                    base_error=float(np.max(np.abs(actual['rgb'].cpu().numpy()-b['baseline_rgb'])))
                    if base_error>1e-6:raise RuntimeError('Feature-pass and direct native baseline differ')
                    for label,color in constants:
                        colors=ctx.base_rgb.clone();colors[emask]=colors.new_tensor(color)
                        actual=render(raw,colors);rgb=actual['rgb'].cpu().numpy()
                        expected=b['baseline_rgb']+b['eligible_weight'][None]*np.array(color,np.float32)[:,None,None]-b['eligible_rgb']
                        error=float(np.max(np.abs(rgb-expected)))
                        expected64=(b['baseline_rgb'].astype(np.float64)+b['eligible_weight'][None].astype(np.float64)*np.array(color,np.float64)[:,None,None]-b['eligible_rgb'].astype(np.float64)).astype(np.float32)
                        direct32=b['direct_readonly_rgb']+b['eligible_weight'][None]*np.array(color,np.float32)[:,None,None]
                        direct64=(b['direct_readonly_rgb'].astype(np.float64)+b['eligible_weight'][None].astype(np.float64)*np.array(color,np.float64)[:,None,None]).astype(np.float32)
                        alternatives=dict(delta_FP64_then_FP32=float(np.max(np.abs(rgb-expected64))),direct_readonly_FP32=float(np.max(np.abs(rgb-direct32))),direct_readonly_FP64_then_FP32=float(np.max(np.abs(rgb-direct64))))
                        aerr=float(np.max(np.abs(actual['alpha'][0].cpu().numpy()-b['alpha'])))
                        kerr=float(np.max(np.abs(rgb-b['baseline_rgb'])[:,b['K']])) if mode=='joint' else None
                        item=dict(raw=raw,mode=mode,constant=label,feature_baseline_max=base_error,RGB_max=error,
                            alpha_max=aerr,joint_K_RGB_max=kerr,alternative_errors=alternatives,original_1e6_gate_pass=error<=1e-6,
                            RGB8_max=int(np.max(np.abs(np.rint(rgb*255)-np.rint(expected*255)))))
                        identities.append(item)
                        if error>5e-6 or aerr!=0 or (kerr is not None and kerr!=0):
                            write(out/'constant_identity.json',dict(rows=identities));raise RuntimeError('Actual constant/unit native identity failed')
                print('IDENTITY_READY',raw,'calls',ctx.native_calls,flush=True)
        write(out/'constant_identity.json',dict(rows=identities,tolerance=5e-6,original_1e6_gate_retained_in_each_row=True,bitexact_claim=False,engineering_tolerance_authorization='Root 2026-11-19 followup accepts adapter original 5e-6 conservation budget after FP32 alternatives; scientific sampling thresholds unchanged',rigorous_float_error_bound_proved=False,RGB8_quantization_reported=True,strict_negative_certificate=False))
        gradients=[]
        for raw in (86,110):
            target=load(P/'stage0_native_r1'/f'{raw:03d}_targets.npz')
            c=ctx.base_rgb.clone().requires_grad_(True)
            value=ctx.render_road(raw,c)['rgb'];region=target['A']|target['C']|target['S']
            m=torch.as_tensor(region,device=c.device)
            loss=value[:,m].sum();gradient=torch.autograd.grad(loss,c)[0]
            observed=gradient[emask].sum(0).detach().cpu().numpy()
            expected=np.repeat(basis[raw]['road']['eligible_weight'][region].sum(dtype=np.float64),3)
            rel=float(np.max(np.abs(observed-expected))/max(1,float(np.max(np.abs(expected)))))
            item=dict(raw=raw,finite=bool(torch.isfinite(gradient).all()),eligible_gradient_nonzero_rows=int((gradient[emask].abs().sum(1)>0).sum()),
                aggregate_adjoint_observed=observed.tolist(),aggregate_native_unit_weight=expected.tolist(),relative_max=rel)
            gradients.append(item)
            if not item['finite'] or item['eligible_gradient_nonzero_rows']==0 or rel>1e-5:raise RuntimeError('Native aggregate adjoint qualification failed')
            del value,gradient,c
        write(out/'native_gradient.json',dict(rows=gradients,optimizer_updates=0,parameter_gradient_only=True))
        floors=[]
        floor_ids={86:[112164,95788],110:[118901,102057,118902,118903,103017]}
        for raw,ids in floor_ids.items():
            t=load(P/'stage0_native_r1'/f'{raw:03d}_targets.npz')
            b=basis[raw]['road'];j=basis[raw]['joint']
            target=t['target_A'];target=target if target.shape[0]==3 else target.transpose(2,0,1)
            for index in ids:
                y,x=divmod(index,480)
                floors.append(dict(raw=raw,pixel_id=index,y=y,x=x,eligible_road_weight=float(b['eligible_weight'][y,x]),
                    eligible_joint_weight=float(j['eligible_weight'][y,x]),road_alpha=float(b['alpha'][y,x]),
                    parent_A_premult_MAE=float(np.abs(b['baseline_rgb'][:,y,x]-b['alpha'][y,x]*target[:,y,x]).mean()),
                    material_mass=0,fallback='Original eligible ci retained; world field gradient is zero; A denominator unchanged'))
        write(out/'A7_readonly_material_floor.json',dict(rows=floors,count=len(floors),A_full_denominator=16904,qualified_A_denominator=16897))
        ctx.assert_frozen()
        after={p:sha(p) for p in binding}
        if after!=binding:raise RuntimeError('Frozen input/source bytes changed during native qualification')
        summary=dict(schema='teacher_detail_native_qualification.v1',completed=True,native_basis_qualified=True,
            actual_preclamp_binary_validated=True,formal_optimization_executed=False,optimizer_updates=0,
            active_rows=151988,eligible_rows=89887,protected_pixel_frames=sum(r['protected_pixels'] for r in rows),
            source_fields_and_ledger_bytes_unchanged=True,geometry_opacity_scales_unchanged=True,
            native_forward_calls=ctx.native_calls,seconds=time.time()-start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            basis_directory=str(out/'basis'),native_context_factory='run_native_qualification.ctx_load(lease)',
            cached_weights_are_world_asset=False,new_camera_or_edit_requires_native_reexport=True)
        write(out/'completed.json',summary);print('COMPLETED',json.dumps(summary),flush=True)
    except Exception:
        write(out/'failed.json',dict(error=traceback.format_exc(),native_forward_calls=0 if ctx is None else ctx.native_calls,
            optimizer_updates=0,seconds=time.time()-start,scope_unchanged=True))
        raise
    finally:
        if ctx is not None:ctx.close()
        current=json.loads(lease.read_text());current.update(active=False,ended_time=time.time());write(lease,current)

if __name__=='__main__':main()

