"""R15-P01/P02：只在N02通过后执行；无训练参数更新。"""
import argparse,gc,json,statistics,sys,time
from pathlib import Path
import torch
from packed_shadow import install


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--base',required=True)
    ap.add_argument('--numeric',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();sys.path.insert(0,args.source)
    gate=json.loads(Path(args.numeric).read_text());choice=gate.get('selected_dtype')
    if not choice:raise RuntimeError('N02未通过，禁止进入性能测试')
    dtype={'torch.float64':torch.float64,'torch.float32':torch.float32}[choice]
    from shadow_rebuild.config import Config,LossWeights
    from shadow_rebuild.data import Dataset
    from shadow_rebuild.model import SceneModel
    from shadow_rebuild.objective import objective
    from shadow_rebuild.train import seed_from_state
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    result=dict(run_id='R15-P01-P02',status='RUNNING',dtype=choice,production_updates=0,
                cases=[],gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__)
    def save():
        p=out/'performance_gate.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False));tmp.replace(p)
    base=Path(args.base);saved=torch.load(base/'paused_checkpoint_004000_20261017/checkpoint.pt',map_location='cpu',weights_only=False)
    cfgraw=saved['config'].copy();cfgraw['weights']=LossWeights(**cfgraw['weights']);cfg=Config(**cfgraw).validate()
    data=Dataset(base/'data','cuda',True);assert saved['input_contract']==data.manifest
    model=SceneModel(seed_from_state(saved['state']),cfg).cuda();model.load_state_dict(saved['state']);model.set_stage('geometry')
    gen=torch.Generator().manual_seed(0)
    def measure(label,raw_id,repeats=2,warmup=False,stage='geometry',optimizer=None):
        index=next(i for i,r in enumerate(data.rows) if int(r['frame_id'])%1000==raw_id)
        samples=[];model.set_stage(stage)
        frame=data.frame(index)
        for i in range(repeats+int(warmup)):
            model.zero_grad(set_to_none=True);gc.collect()
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            if optimizer is not None:frame=data.frame(index)
            loss,terms,package=objective(model,frame,len(data.actor_ids),'cuda',stage,gen,None,4000)
            torch.cuda.synchronize();forward=time.perf_counter()-start
            loss.backward();torch.cuda.synchronize();backward=time.perf_counter()-start-forward
            extra=0
            if optimizer is not None:
                e=time.perf_counter();torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True)
                optimizer.step();torch.cuda.synchronize();extra=time.perf_counter()-e
            sample=dict(forward=forward,backward=backward,total=time.perf_counter()-start,
                optimizer_extra=extra,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,loss=float(loss.detach()))
            if not warmup or i:samples.append(sample)
            del loss,terms,package
        record=dict(variant=label,raw=raw_id,stage=stage,samples=samples,
            median_seconds=statistics.median(x['total'] for x in samples),
            peak_allocated_GiB=max(x['peak_allocated_GiB'] for x in samples))
        result['cases'].append(record);save();print(json.dumps(record),flush=True)
        model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();return record
    try:
        measure('original_shadow',107,2)
        cfg.shadows=False;measure('shadow_disabled',107,2);cfg.shadows=True
        install(model,dtype=dtype,block_size=256,pair_batch=256,recompute=True)
        for raw_id in [93,105,107,117,121,137]:measure('packed_factors',raw_id,3,True)
        packed=[c for c in result['cases'] if c['variant']=='packed_factors']
        median=statistics.median(c['median_seconds'] for c in packed)
        worst=max(c['median_seconds'] for c in packed);memory=max(c['peak_allocated_GiB'] for c in packed)
        result['geometry_gate']=dict(cross_frame_median_seconds=median,worst_frame_median_seconds=worst,
            peak_allocated_GiB=memory,passed=median<=4 and worst<=6 and memory<=14)
        save()
        # 阶段计时保留本来的接收网格；teacher不在4k checkpoint，单列限制。
        if result['geometry_gate']['passed']:
            for stage in ['material','joint']:measure('packed_stage_probe_no_teacher',107,1,False,stage)
            result['stage_probe_limit']='frozen4k capacity; missing later body teacher, not actual future training throughput'
            model.set_stage('geometry');optimizer=model.optimizer('geometry');optimizer.load_state_dict(saved['optimizer'])
            for group in optimizer.param_groups:group['lr']=0.
            measure('packed_end_to_end_lr0_Adam',107,2,False,'geometry',optimizer)
            result['optimizer_probe']='original Adam moments loaded; lr=0, no schedule/project/topology, only disposable moments updated'
        result.update(status='COMPLETE',human_verdict=None,failure_ledger_refs=['V82-F05'])
        save()
    except BaseException as e:
        result.update(status='ERROR',error=repr(e));save();raise


if __name__=='__main__':main()
