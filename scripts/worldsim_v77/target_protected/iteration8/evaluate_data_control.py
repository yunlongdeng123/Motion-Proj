"""冻结两套输入，三权重独立加载；真实DELETE无actor-free GT。"""
from pathlib import Path
import sys,json,os,time,signal,fcntl,gc
import numpy as np
from PIL import Image
P=Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected')
sys.path.insert(0,str(P.parent));sys.path.insert(0,str(P/'iteration7'));sys.path.insert(0,str(Path(__file__).parent))
from asset_factory import O,T,read,dump
from audit import protected_mask
from evaluate_control import metrics

def freeze():
    path=O/'evaluation_plan.json'
    if path.exists():return read(path)
    catalog=read(O/'dataset_catalog.json');cases=[]
    for c in catalog['cases']:
        if c['split']=='validation':cases.append(dict(c,eval_id='synthetic_'+c['case_id'],kind='synthetic',frames=list(range(10)),role='synthetic_validation',GT_available=True))
    cases+=read(O/'real_evaluation_plan.json')['cases']
    assert cases and any(c['kind']=='synthetic' for c in cases)
    assert all(c.get('input_valid',True) for c in cases)
    plan={'cases':cases,'arms':['base','r7','new_data'],'seed':42,'steps':25,'frames':10,'inference_resolution':[576,1024],
          'scene_split':read(O/'evaluation_source_split.json'),'same_input_for_every_arm':True,'synthetic_GT':'true nuScenes train RGB untouched; held receiver scenes are training-world sources, not official final test','real_GT':None,'real_scope':'existing exposed DEV badcases; hidden identities cannot be quantified by RGB GT MAE','human_verdict':None}
    dump(path,plan);return plan

def inputs(c):
    f=Path(c['folder']);ids=c['frames']
    if c['kind']=='synthetic':
        load=lambda r,rgb:[np.asarray(Image.open(f/r/f'{i:03}.png').convert('RGB' if rgb else 'L')) for i in ids]
        x=load('X',True);y=load('Y',True);h=[m>0 for m in load('model_hole',False)]
        b=[protected_mask(f,i,(576,1024),c['type']) for i in ids];alpha=[v.astype('float32') for v in h]
    else:
        def load(r,ext,rgb):
            paths=sorted((f/r).glob('*.'+ext));return [np.asarray(Image.open(paths[i]).convert('RGB' if rgb else 'L')) for i in ids]
        x=load('rgb','jpg',True);y=None;h=[v>0 for v in load('model_mask','png',False)];b=[v>0 for v in load('protect','png',False)];alpha=[v.astype('float32')/255 for v in load('alpha','png',False)]
    assert len(x)==10 and all(v.shape==(576,1024,3) for v in x) and all(v.any() for v in h)
    assert all(not np.any((a>0)&~m) for a,m in zip(alpha,h))
    return x,y,h,b,alpha

def main(arms,suite='all'):
    plan=read(O/'real_evaluation_plan.json') if suite=='real' else freeze();dest=O/'evaluation';dest.mkdir(exist_ok=True)
    lock=open(dest/'evaluate.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=read(dest/'state.json') if (dest/'state.json').exists() else {'completed':[]}
    done={(r['eval_id'],r['arm']) for r in state['completed']}
    from repair_drive import Engine,set_seed,torch
    from safetensors.torch import load_file
    torch.set_num_threads(4);e=None
    def timeout(*_):raise TimeoutError('fixed native inference window exceeded300sec')
    signal.signal(signal.SIGALRM,timeout)
    for arm in arms:
        if {(c['eval_id'],arm) for c in plan['cases']}<=done:continue
        if e is not None:del e;gc.collect();torch.cuda.empty_cache()
        e=Engine() # 每一臂重新恢复原权重，不让上一个patch污染本臂。
        if arm!='base':
            folder=T/'r7/encoder_fixed_lowres/training' if arm=='r7' else O/'training'
            assert read(folder/'state.json')['stage']=='complete' and read(folder/'state.json')['steps']==160
            patch=load_file(str(folder/'attention_step_0160.safetensors'));cfg=read(folder/'config.json')
            assert len(patch)==80 and set(patch)==set(cfg['trainable_tensors'])
            _,unexpected=e.model.load_state_dict(patch,strict=False);assert not unexpected
            dump(dest/f'{arm}_patch_loaded.json',{'path':str(folder/'attention_step_0160.safetensors'),'keys_loaded':len(patch),'architecture_unchanged':True,'fresh_original_model':True});del patch
        for c in plan['cases']:
            cid=c['eval_id'];out=dest/cid
            if (cid,arm) in done:
                assert len(list((out/arm).glob('*.png')))==10 and len(list((out/(arm+'_native')).glob('*.png')))==10
                continue
            out.mkdir(exist_ok=True);x,y,h,b,alphas=inputs(c)
            for r in ['input','GT','condition','mask',arm,arm+'_native']:(out/r).mkdir(exist_ok=True)
            previous=T/'r6/evaluation'/cid
            reused=arm=='base' and c['kind']=='real_development' and (previous/'base_metrics.json').exists()
            if reused:
                meta=read(previous/'base_metrics.json');oldplan=read(T/'r6/evaluation_plan.json');oldcase=next((v for v in oldplan['cases'] if v['eval_id']==cid),None)
                # A041_w10在补充plan登记，原空窗不能复用。
                if oldcase is None:reused=False
                else:
                    assert oldcase['frames']==c['frames'] and oldplan['seed']==42 and oldplan['steps']==25 and meta['previous_condition'] is False
                    assert all(np.array_equal(xx,np.asarray(Image.open(previous/'input'/f'{i:05}.png'))) and np.array_equal(hh,np.asarray(Image.open(previous/'mask'/f'{i:05}.png'))>0) for i,(xx,hh) in enumerate(zip(x,h)))
            torch.cuda.reset_peak_memory_stats();start=time.monotonic()
            if reused:raws=[np.asarray(Image.open(previous/'base_native'/f'{i:05}.png')) for i in range(10)]
            else:
                e.im=x;e.masks=h;e.previous_segment_last_frame=None;e.im_result=[];set_seed(42);signal.alarm(300)
                try:e.predict(1,False,'Deletion')
                finally:signal.alarm(0)
                raws=e.im_result
            assert len(raws)==10
            scores=[]
            for i,(xx,hh,bb,alpha,raw) in enumerate(zip(x,h,b,alphas,raws)):
                comp=np.rint(raw*alpha[...,None]+xx*(1-alpha[...,None])).clip(0,255).astype('uint8')
                assert np.array_equal(comp[~hh],xx[~hh])
                Image.fromarray(comp).save(out/arm/f'{i:05}.png');Image.fromarray(raw).save(out/(arm+'_native')/f'{i:05}.png')
                if arm=='base':
                    Image.fromarray(xx).save(out/'input'/f'{i:05}.png');Image.fromarray(hh.astype('uint8')*255).save(out/'mask'/f'{i:05}.png')
                    cc=xx.copy();cc[hh]=127;Image.fromarray(cc).save(out/'condition'/f'{i:05}.png')
                    if y is not None:Image.fromarray(y[i]).save(out/'GT'/f'{i:05}.png')
                if y is not None:scores.append(dict(frame=i,**metrics(raw,y[i],hh,bb)))
                else:scores.append({'frame':i,'actor_free_GT_available':False,'RGB_recovery_MAE':None,'outside_write_mask_pixels_changed':int((comp[~hh]!=xx[~hh]).any(-1).sum()),'protected_region_pixels':int(bb.sum()),'protected_region_pixels_overlapped_by_generation_mask':int((bb&hh).sum())})
            row={'eval_id':cid,'arm':arm,'kind':c['kind'],'source':c,'scores':scores,'seconds':time.monotonic()-start,'seed':42,'steps':25,'previous_condition':False,'reused_from':str(previous) if reused else None,'peak_allocated_GiB':None if reused else torch.cuda.max_memory_allocated()/2**30,'human_verdict':None}
            dump(out/f'{arm}_metrics.json',row);state['completed'].append({k:row[k] for k in ['eval_id','arm','kind','seconds']});done.add((cid,arm));state.update(stage=arm,completed_count=len(done),total=len(plan['cases'])*len(plan['arms']));dump(dest/'state.json',state);print(json.dumps(state['completed'][-1]),flush=True)
    state['stage']='complete_requested_arms';state['arms_requested']=arms;state['all_three_arms_complete']={(c['eval_id'],a) for c in plan['cases'] for a in plan['arms']}<=done;dump(dest/'state.json',state)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--arms',nargs='+',choices=['base','r7','new_data'],default=['base','r7','new_data']);p.add_argument('--plan-only',action='store_true');p.add_argument('--suite',choices=['real','all'],default='all');a=p.parse_args()
    if a.plan_only:freeze()
    else:main(a.arms,a.suite)
