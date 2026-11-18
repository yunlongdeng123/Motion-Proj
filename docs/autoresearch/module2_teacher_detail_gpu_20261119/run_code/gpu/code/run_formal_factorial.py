"""Fixed-budget teacher-HF/reader factorial; all scene geometry remains frozen."""
from pathlib import Path
import argparse,hashlib,json,random,sys,time,traceback
import numpy as np
import torch
from PIL import Image

Q=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_teacher_detail_cpu_20261119_r1')
RUN=Q/'gpu_20261119';F=RUN/'field_training'
sys.path[:0]=[str(F/'code'),str(Q/'code')]
from training_factory import default_config,prepare_inputs,make_state,load_native_adapter
from field_training_state import cpu_clone,max_tree_difference

FRAMES=(86,110,134,179,198)
def write(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def export(state,prepared,directory):
    directory.mkdir(parents=True,exist_ok=False);rows=[]
    with torch.no_grad():
        for raw in FRAMES:
            packet=state.reader.render(state.model,raw,joint=True)
            arrays={k:packet[k].detach().cpu().numpy() for k in ('road_rgb','road_alpha','joint_rgb','joint_alpha')}
            arrays['K']=prepared.basis[raw]['joint']['K']
            for key in ('reader_source_mass','fallback_mask','eligible_weight_road','eligible_weight_joint'):
                if key in packet:arrays[key]=packet[key].detach().cpu().numpy()
            target=prepared.basis[raw]
            kerr=float(np.abs(arrays['joint_rgb']-target['joint']['baseline_rgb'])[:,arrays['K']].max())
            aerr=max(float(np.abs(arrays[k+'_alpha']-target[k]['alpha']).max()) for k in ('road','joint'))
            if kerr!=0 or aerr!=0:raise RuntimeError(('Fixed K/coverage changed',raw,kerr,aerr))
            np.savez_compressed(directory/('%03d.npz'%raw),**arrays)
            for mode in ('road','joint'):
                image=np.rint(np.clip(arrays[mode+'_rgb'].transpose(1,2,0),0,1)*255).astype(np.uint8)
                Image.fromarray(image).save(directory/('%03d_%s.png'%(raw,mode)))
            row=dict(raw=raw,joint_K_max=kerr,alpha_max=aerr,K_pixels=int(arrays['K'].sum()))
            if raw in (86,110):
                t=prepared.targets[raw];alpha=t['fixed_alpha'].detach().cpu().numpy()
                row['MAE']={k:float(np.abs(arrays['road_rgb']-alpha[None]*t['target_'+k].detach().cpu().numpy())[:,t[k].cpu().numpy()].mean()) for k in ('A','C','S')}
            rows.append(row)
    assert sum(r['K_pixels'] for r in rows)==77266
    write(directory/'evaluation.json',dict(rows=rows,reader=state.reader.kind,arm=state.arm,cursor=state.cursor))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    start=time.time();physical=0;completed=[];adapter=None
    smoke=json.loads((RUN/'lifecycle_r1/receipt.json').read_text())
    assert smoke['verified'] and smoke['native_GPU_executed'] and not smoke['hour_scale_ten_minute_required']
    launch=RUN/'formal_launch_contract.json'
    contract=dict(schema='teacher_detail_formal_launch.v1',formal_authorized=True,root_accepted=True,
        user_authorization=str(RUN/'authorization.json'),scientific_protocol=str(Q/'protocol.json'),
        real_smoke_receipt=str(RUN/'lifecycle_r1/receipt.json'),actual_smoke_updates=17,
        native_qualification=str(RUN/'native_qualification_r2/completed.json'),
        source=__file__,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        steps=dict(A=200,B=300,C=300),physical_budget=3400,logical_budget=4000,
        all_geometry_coverage_and_source_ledgers_frozen=True,high_bound=.34,GPU_keep_on=True,
        execution_order=['common_pixel','M11','M01','T11','common_avg','M10','M00'],
        smoke_formal_start='fresh zero L/H; no smoke state reused',
        timing='Measured conservative queue <1h. No hour-scale ten-minute requirement; short full lifecycle completed.')
    assert not launch.exists(),'Preserve prior launch contract'
    write(launch,contract)
    lease=F/'gpu_lease.json';l=json.loads(lease.read_text());l.update(phase='formal',formal_allowed=True);write(lease,l)
    config=default_config();config['formal_launch_contract']=str(launch)
    write(out/'started.json',dict(time=start,PID=__import__('os').getpid(),physical_updates=0,contract=contract))
    torch.set_num_threads(8)
    try:
        prepared=prepare_inputs(config,device='cuda');adapter=load_native_adapter(config)
        write(out/'input_bindings.json',prepared.bindings)
        seed=np.load(config['T11_seed'])['H_seed_latent']
        for kind,arms in [('pixel',[('M11','teacher',False),('M01','observed',False),('T11','teacher',True)]),
                          ('avg',[('M10','teacher',False),('M00','observed',False)])]:
            torch.manual_seed(101);torch.cuda.manual_seed_all(101);np.random.seed(101);random.seed(101)
            common=make_state(kind,'observed','formal','common_'+kind,prepared=prepared,adapter=adapter,config=config)
            export(common,prepared,out/(kind+'_initial')/'rendered')
            trace=[]
            for _ in range(200):
                r=common.step();physical+=1;trace.append(r)
                if common.cursor%50==0:print(json.dumps(dict(arm=common.arm,physical=physical,**r)),flush=True)
            a_dir=common.save(out/(kind+'_common_A200'));A=common.snapshot()
            export(common,prepared,a_dir/'rendered');write(a_dir/'optimization.json',trace)
            for arm,gate,seeded in arms:
                state=make_state(kind,gate,'formal',arm,prepared=prepared,adapter=adapter,config=config)
                state.fork_common_A(A,high_seed=seed if seeded else None)
                beforeL=state.model.low.detach().clone();adamL=cpu_clone(state.optimizer.state[state.model.low])
                branch_trace=[];checkpoints=[]
                for end in (500,800):
                    while state.cursor<end:
                        r=state.step();physical+=1;branch_trace.append(r)
                        if state.cursor%50==0:
                            print(json.dumps(dict(arm=arm,physical=physical,**r)),flush=True)
                            write(out/'progress.json',dict(arm=arm,cursor=state.cursor,physical_updates=physical,completed=completed,seconds=time.time()-start))
                    if end==500:
                        assert torch.equal(beforeL,state.model.low)
                        assert max_tree_difference(adamL,cpu_clone(state.optimizer.state[state.model.low]))==0
                    directory=state.save(out/arm/('B500'if end==500 else'C800'))
                    checks=export(state,prepared,directory/'rendered');checkpoints.append(str(directory))
                    write(directory/'optimization.json',branch_trace)
                assert state.trace.count(86)==state.trace.count(110)==400
                completed.append(dict(arm=arm,reader=kind,gate=gate,T11_H_seed=seeded,
                    logical_updates=800,physical_branch_updates=600,checkpoints=checkpoints))
                write(out/'completed_arms.json',completed)
                print('ARM_COMPLETED',arm,'physical',physical,flush=True)
                del state,beforeL,adamL
            del common
        adapter.assert_frozen()
        for path,digest in prepared.bindings.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise RuntimeError('Bound input/source modified: '+path)
        assert physical==3400 and len(completed)==5
        receipt=dict(schema='teacher_detail_formal_completed.v1',completed=True,arms=completed,
            formal_physical_updates=physical,formal_logical_updates=4000,engineering_updates_separate=17,
            seconds=time.time()-start,native_forward_calls=adapter.ctx.native_calls,
            scene_geometric_parameters_trainable=False,world_material_low_high_only=True,
            all_bound_inputs_unchanged=True,joint_K_pixels=77266,joint_K_max=0,alpha_max=0,
            GPU_keep_on=True,power_operations=0,production_promoted=False,human_verdict=None,problem_solved=None)
        write(out/'completed.json',receipt);print('COMPLETED',json.dumps(receipt),flush=True)
    except Exception:
        write(out/'failed.json',dict(error=traceback.format_exc(),physical_updates=physical,completed=completed,seconds=time.time()-start))
        raise
    finally:
        if adapter is not None:adapter.ctx.close()

if __name__=='__main__':main()
