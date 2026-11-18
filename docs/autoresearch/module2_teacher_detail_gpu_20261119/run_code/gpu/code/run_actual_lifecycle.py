"""Real-GPU short lifecycle; engineering updates never seed formal training."""
from pathlib import Path
import copy,json,sys,time,traceback
import numpy as np
import torch

Q=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1')
RUN=Q/'gpu_20261119';F=RUN/'field_training'
sys.path[:0]=[str(F/'code'),str(Q/'code')]
from training_factory import prepare_inputs,make_state,load_native_adapter
from field_training_state import max_tree_difference,cpu_clone
from world_texture_runtime import WorldTextureRuntime

def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def torchload(path):return torch.load(path,map_location='cpu',weights_only=False)

def render_tensor(state,raw,joint=False):
    packet=state.reader.render(state.model,raw,joint=joint)
    key='joint_rgb' if joint else 'road_rgb'
    return packet[key] if key in packet else packet['road_rgb']

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    started=time.time();write(out/'started.json',dict(PID=__import__('os').getpid(),role='smoke',time=started))
    assert json.loads((RUN/'authorization.json').read_text())['GPU_authorized']
    assert json.loads((RUN/'native_qualification_r2/completed.json').read_text())['native_basis_qualified']
    torch.set_num_threads(8)
    adapter=None;updates=0;rows=[];traces=[]
    try:
        prepared=prepare_inputs(device='cuda')
        adapter=load_native_adapter()
        for kind,arm in [('avg','M10'),('pixel','M11')]:
            common=make_state(kind,'observed','smoke','common_'+kind,prepared=prepared,adapter=adapter,stage_counts=(2,2,2))
            for _ in range(2):traces.append(common.step());updates+=1
            saved=common.save(out/(kind+'_common_A2'))
            a=torchload(saved/'checkpoint.pt')
            # Continue and replay the identical B step with full Adam/RNG/sampler restoration.
            common.step();updates+=1;expected=common.snapshot()
            common.restore(a);restored=common.snapshot()
            before_diff=max_tree_difference(a,restored)
            common.step();updates+=1;after_diff=max_tree_difference(expected,common.snapshot())
            if before_diff!=0 or after_diff>1e-6:raise RuntimeError(('Full restore/replay failed',kind,before_diff,after_diff))
            branch=make_state(kind,'teacher','smoke',arm,prepared=prepared,adapter=adapter,stage_counts=(2,2,2))
            original_gate=branch.model.H_gate.clone()
            branch.fork_common_A(a)
            assert torch.equal(original_gate,branch.model.H_gate)
            low_A=branch.model.low.detach().clone();adamL=cpu_clone(branch.optimizer.state[branch.model.low])
            gradB=[]
            for _ in range(2):
                r=branch.step();updates+=1;traces.append(r);gradB.append(r['gradients']['high'])
            assert torch.equal(low_A,branch.model.low)
            assert max_tree_difference(adamL,cpu_clone(branch.optimizer.state[branch.model.low]))==0
            if min(gradB)<=0:raise RuntimeError('Teacher high-frequency field has no gradient')
            for _ in range(2):traces.append(branch.step());updates+=1
            final=branch.save(out/(kind+'_ABC6'))
            with torch.no_grad():
                road_before={raw:render_tensor(branch,raw).detach().clone() for raw in (86,110,134,179,198)}
                joint_before={raw:render_tensor(branch,raw,True).detach().clone() for raw in (86,110,134,179,198)}
            loaded,meta=WorldTextureRuntime.load(final/'field.pt',device='cuda')
            field_max=0.;Kmax=0.;Kcount=0
            with torch.no_grad():
                for raw in (86,110,134,179,198):
                    packet=branch.reader.render(loaded,raw,joint=False)
                    field_max=max(field_max,float((packet['road_rgb']-road_before[raw]).abs().max()))
                    packet=branch.reader.render(loaded,raw,joint=True)
                    rgb=packet.get('joint_rgb',packet.get('road_rgb'))
                    field_max=max(field_max,float((rgb-joint_before[raw]).abs().max()))
                    basis=prepared.basis[raw]['joint']
                    mask=torch.as_tensor(basis['K'],device=rgb.device,dtype=torch.bool)
                    parent=torch.as_tensor(basis['baseline_rgb'],device=rgb.device)
                    Kcount+=int(mask.sum());Kmax=max(Kmax,float((rgb-parent)[:,mask].abs().max()))
            if field_max>1e-6 or Kmax!=0 or Kcount!=77266:raise RuntimeError(('Persistence/K failed',field_max,Kmax,Kcount))
            rows.append(dict(reader=kind,full_restore_before_step_max=before_diff,next_step_full_state_max=after_diff,
                B_low_and_AdamL_exact=True,teacher_high_gradient_max=gradB,H_gate_preserved_on_fork=True,
                field_reload_render_max=field_max,joint_K_max=Kmax,joint_K_pixels=Kcount,
                seconds=time.time()-started,physical_updates=updates))
            write(out/'progress.json',dict(rows=rows,physical_updates=updates))
            print(json.dumps(rows[-1]),flush=True)
            del loaded,branch,common,road_before,joint_before
        write(out/'trace.json',traces)
        # A fork plus bounded HF seed must leave low, base, Adam and frozen ledgers intact.
        common=make_state('pixel','observed','smoke','common_pixel',prepared=prepared,adapter=adapter,stage_counts=(2,2,2))
        common.restore(torchload(out/'pixel_common_A2/checkpoint.pt'))
        seeded=make_state('pixel','teacher','smoke','T11',prepared=prepared,adapter=adapter,stage_counts=(2,2,2))
        seed=np.load(Q/'trellis_hf/seed_r5_nodal_b034/H_seed.npz')['H_seed_latent']
        seeded.fork_common_A(common.snapshot(),high_seed=seed)
        if max_tree_difference(common.snapshot()['optimizer'],seeded.snapshot()['optimizer'])!=0:raise RuntimeError('Seed changed optimizer')
        seeded.step();updates+=1
        assert torch.equal(common.model.low,seeded.model.low)
        adapter.assert_frozen()
        seconds=time.time()-started
        # Real timing decides whether the hour-scale project's ten-minute rule applies.
        max_step=max(t['elapsed_seconds'] for t in traces)
        estimate_seconds=3400*max_step+300
        receipt=dict(schema='teacher_detail_real_GPU_lifecycle.v1',verified=True,native_GPU_executed=True,
            rows=rows,role='smoke',formal_updates=0,engineering_optimizer_updates=updates,
            T11_seed_only_and_next_B_step=True,seconds=seconds,
            steady_step_seconds_max=max_step,conservative_formal_queue_estimate_seconds=estimate_seconds,
            hour_scale_ten_minute_required=estimate_seconds>=3600,
            full_state_restore_and_stage_entries=True,field_persistence=True,K_exact=True,
            geometric_and_source_ledgers_frozen=True,power_operations=0,GPU_keep_on=True)
        if receipt['hour_scale_ten_minute_required'] and seconds<600:
            receipt['verified']=False;receipt['requires_ten_minute_extension']=True
        write(out/'receipt.json',receipt);print(json.dumps(receipt),flush=True)
    except Exception:
        write(out/'failed.json',dict(error=traceback.format_exc(),engineering_optimizer_updates=updates,seconds=time.time()-started))
        raise
    finally:
        if adapter is not None and hasattr(adapter,'ctx'):adapter.ctx.close()

if __name__=='__main__':main()
