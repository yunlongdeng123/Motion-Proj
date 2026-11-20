"""Two necessary RGB601 engineering replays with separate numeric units."""
from pathlib import Path
import copy,json,sys,time,traceback
import numpy as np
C=Path('/root/autodl-tmp/drivingeditor_repro/campaign');Q=C/'road_teacher_detail_cpu_20261119_r1';B=Q/'gpu_20261119/native_capacity_20261120_r1'
P=B/'smoke_r1';M=P/'missing_paths_r1';OUT=P/'RGB_replay_audit_r1';OWNER='native_capacity_smoke';FRAMES=(86,110)
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
    OUT.mkdir(parents=True,exist_ok=False);failure=read(M/'failed.json');ledger=copy.deepcopy(failure['aggregate_ledger']);start=time.monotonic();ctx=None
    write(OUT/'authorization.json',dict(GPU_authorized=True,formal_allowed=False,root_authorization='Necessary RGB601 per-tensor restore/replay diagnosis; no600 rerun',additional_updates_cap=2,aggregate_updates_cap=4000))
    lease=OUT/'gpu_lease.json';write(lease,dict(gpu_authorized=True,active=True,owner=OWNER,authorization=str(OUT/'authorization.json')))
    try:
        import torch
        torch.set_num_threads(1)
        sys.path[:0]=[str(B/'code'),str(B/'shared_field/code')]
        from native_capacity_inputs import make_context,field_and_reader,real_frames
        from native_capacity_loss import NativeCapacityLoss
        from shared_training_state import SharedTrainingState
        from shared_field_gs_read import file_sha
        saved=torch.load(M/'RGB600_checkpoint.pt',map_location='cpu',weights_only=False)
        frames,_=real_frames('seen');ctx=make_context(lease,OWNER);model,reader,metadata=field_and_reader(ctx.base_rgb,'cuda')
        loss=NativeCapacityLoss(frames,device='cuda');loss.restore(saved['objective'])
        def road(raw,color):
            if ledger['native_forwards']>=20000:raise RuntimeError('F cap')
            ledger['native_forwards']+=1;return ctx.render_road(raw,color)
        class View:
            frames=FRAMES;active_world_rows=ctx.active_world_rows
            @property
            def native_calls(self):return ctx.native_calls
            def render_road(self,raw,color):return road(raw,color)
        def hook(e):
            if e['event']=='native_backward_calls':ledger['native_per_view_backwards']+=e['amount']
            elif e['event']=='backward_calls':ledger['torch_backward_invocations']+=e['amount']
            elif e['event']=='parameter_updates':ledger['engineering_accepted_updates']+=1;ledger['shared_updates']+=1;ledger['replay_updates']+=1
        state=SharedTrainingState(model,reader,View(),loss,mode='rgb',source_identity=saved['source_identity'],source_metadata=saved['source_metadata'],purpose='smoke',ledger_hook=hook)
        def same(a,b):
            if torch.is_tensor(a):return torch.equal(a,b)
            if isinstance(a,np.ndarray):return np.array_equal(a,b)
            if isinstance(a,dict):return a.keys()==b.keys()and all(same(a[k],b[k])for k in a if k!='physical')
            if isinstance(a,(tuple,list)):return len(a)==len(b)and all(same(x,y)for x,y in zip(a,b))
            return a==b
        state.restore(saved);restore_a=same(saved,state.snapshot());a_step=state.step();a=state.snapshot();torch.save(a,OUT/'601_reference.pt')
        with torch.no_grad():ca=reader.colors(model).detach().clone();pa={r:road(r,ca)for r in FRAMES}
        state.restore(saved);restore_b=same(saved,state.snapshot());b_step=state.step();b=state.snapshot();torch.save(b,OUT/'601_replay.pt')
        with torch.no_grad():cb=reader.colors(model).detach().clone();pb={r:road(r,cb)for r in FRAMES}
        def stats(x,y):
            x=x.detach().double().cpu();y=y.detach().double().cpu();d=x-y;scale=max(float(x.abs().max()),float(y.abs().max()))
            return dict(exact=torch.equal(x,y),max_abs=float(d.abs().max()),RMS=float(d.square().mean().sqrt()),
                reference_max_abs=float(x.abs().max()),replay_max_abs=float(y.abs().max()),relative_Linf=float(d.abs().max())/max(scale,1e-30))
        details={name:stats(a['params'][name],b['params'][name])for name in('low','high')};moments={}
        for j,name in enumerate(('low','high')):
            moments[name]={key:stats(a['optimizer']['state'][j][key],b['optimizer']['state'][j][key])for key in('step','exp_avg','exp_avg_sq')}
        images={str(r):dict(RGB=stats(pa[r]['rgb'],pb[r]['rgb']),alpha_exact=torch.equal(pa[r]['alpha'],pb[r]['alpha']))for r in FRAMES}
        result=dict(schema='RGB601_unit_separated_replay_audit.v1',complete=True,restore_first_exact=restore_a,restore_second_exact=restore_b,
            parameter_details=details,Adam_moments=moments,RNG_exact=same(a['RNG'],b['RNG']),groups_exact=same(a['optimizer']['param_groups'],b['optimizer']['param_groups']),
            cursor_exact=a['cursor']==b['cursor']==601,frozen_buffers_exact=same(a['frozen_buffers'],b['frozen_buffers']),
            ci_RGB=stats(ca,cb),native_road=images,loss_values=[a_step['loss'],b_step['loss']],
            extra_updates=2,extraF=8,extraB=4,aggregate_ledger=ledger,seconds=time.monotonic()-start,
            source_SHA256={str(p):file_sha(p)for p in [Path(__file__),M/'RGB600_checkpoint.pt',B/'shared_field/code/shared_training_state.py']},
            interpretation='Per-unit numeric evidence; no uniform mixed-unit max PASS and no formal warm start')
        write(OUT/'receipt.json',result);print(json.dumps(result),flush=True)
    except Exception:
        write(OUT/'failed.json',dict(error=traceback.format_exc(),aggregate_ledger=ledger,seconds=time.monotonic()-start));raise
    finally:
        if ctx is not None:ctx.close()
        v=read(lease);v.update(active=False,owner=None,ended_at=time.time());write(lease,v)
if __name__=='__main__':main()
