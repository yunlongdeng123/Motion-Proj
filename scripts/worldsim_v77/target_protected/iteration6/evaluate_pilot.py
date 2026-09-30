"""固定4个合成留出例+4个已曝光真实DELETE开发例，原/微调同mask、seed、窗口。"""
from pathlib import Path
import argparse,json,sys,time,os,signal
import numpy as np
import fcntl
from PIL import Image,ImageDraw
ROOT_BASE=Path('/root/autodl-tmp/runs/worldsim_v77')
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77')
sys.path.insert(0,str(Path(__file__).parent))
from train_pilot import save_json,selected

def plan(root):
    catalog=json.loads((root/'dataset_catalog.json').read_text())
    cases=[dict(c,eval_id='synthetic_'+c['case_id'],kind='synthetic',frames=list(range(10)),seed=42,steps=25) for c in catalog['cases'] if c['split']=='validation']
    audit=ROOT_BASE/'WS-V77-DELETE-AUDIT-20260928/r1'
    clips={c['clip_id']:c for c in json.loads((audit/'selection.json').read_text())['clips']}
    for cid in ('A022','A041','A013','A048'):
        cases.append(dict(clips[cid],eval_id=cid,kind='real_development',folder=str(audit/'clips'/cid),frames=list(range(10)),seed=42,steps=25,GT_available=False))
    result={'cases':cases,'checkpoint_step':160,'training_input_size':[320,576],'inference_size':[576,1024],'seed':42,'steps':25,'num_frames':10,'boundary':'known val failures are exposed development examples, not final unseen test; synthetic source scenes held apart from train; no Omega/GLB'}
    save_json(root/'evaluation_plan.json',result);return result

def inputs(c):
    folder=Path(c['folder'])
    def load(role,ext,rgb):
        files=sorted((folder/role).glob('*.'+ext))
        return [np.asarray(Image.open(files[i]).convert('RGB' if rgb else 'L')) for i in c['frames']]
    if c['kind']=='synthetic':
        x=load('X','png',True);y=load('Y','png',True);m=[v>0 for v in load('model_hole','png',False)]
        protected=load('protected','png',False) if (folder/'protected').exists() else [np.zeros(v.shape[:2],np.uint8) for v in x]
        alphas=[v.astype('float32') for v in m]
    else:
        x=load('rgb','jpg',True);y=None;m=[v>0 for v in load('model_mask','png',False)];protected=load('protect','png',False)
        alphas=[v.astype('float32')/255 for v in load('alpha','png',False)]
    assert len(x)==10 and all(v.shape==(576,1024,3) for v in x)
    return x,y,m,protected,alphas

def main(a):
    root=a.root;p=json.loads((root/'evaluation_plan.json').read_text()) if (root/'evaluation_plan.json').exists() else plan(root)
    if a.plan_only:return
    checkpoint=root/'training/attention_step_0160.safetensors'
    assert checkpoint.exists()
    from repair_drive import Engine,set_seed,torch
    from safetensors.torch import load_file
    torch.set_num_threads(4);dest=root/'evaluation';dest.mkdir(exist_ok=True)
    lock=open(dest/'evaluate.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=json.loads((dest/'state.json').read_text()) if (dest/'state.json').exists() else {'stage':'base','completed':[],'human_verdict':None}
    finished={(c['eval_id'],c['arm']) for c in state['completed']}
    if state['stage']=='complete' and {(c['eval_id'],arm) for c in p['cases'] for arm in ('base','finetuned')}<=finished:
        print('ALREADY_COMPLETE: no duplicate inference');return
    e=Engine();save_json(dest/'state.json',state)
    def timeout(*_):raise TimeoutError('single fixed window exceeded 300 sec')
    signal.signal(signal.SIGALRM,timeout)
    for arm in ('base','finetuned'):
        if arm=='finetuned':
            patch=load_file(str(checkpoint));assert len(patch)==80 and all(selected(k) for k in patch)
            missing,unexpected=e.model.load_state_dict(patch,strict=False);assert not unexpected
            save_json(dest/'patch_loaded.json',{'keys_loaded':len(patch),'architecture_unchanged':True,'base_others_retained':len(missing)})
        for c in p['cases']:
            if (c['eval_id'],arm) in finished:
                folder=dest/c['eval_id']
                assert (folder/f'{arm}_metrics.json').exists() and len(list((folder/arm).glob('*.png')))==10 and len(list((folder/(arm+'_native')).glob('*.png')))==10
                continue
            folder=dest/c['eval_id'];folder.mkdir(exist_ok=True)
            x,y,m,protected,alphas=inputs(c)
            if not all(mask.any() for mask in m):raise ValueError('nonempty mask required for every frame; empty windows cannot be successful DELETE')
            for role in ('input','GT','condition','mask',arm,arm+'_native'):(folder/role).mkdir(exist_ok=True)
            e.im=x;e.masks=m;e.previous_segment_last_frame=None;e.im_result=[];set_seed(42)
            torch.cuda.reset_peak_memory_stats();start=time.time();signal.alarm(300)
            try:e.predict(1,False,'Deletion')
            finally:signal.alarm(0)
            assert len(e.im_result)==10
            scores=[]
            for i,(im,mask,raw,alpha,b) in enumerate(zip(x,m,e.im_result,alphas,protected)):
                comp=np.rint(raw*alpha[...,None]+im*(1-alpha[...,None])).astype('uint8')
                assert not np.any(comp[~mask]!=im[~mask])
                Image.fromarray(comp).save(folder/arm/f'{i:05}.png');Image.fromarray(raw).save(folder/(arm+'_native')/f'{i:05}.png')
                if arm=='base':
                    Image.fromarray(im).save(folder/'input'/f'{i:05}.png');Image.fromarray((mask*255).astype('uint8')).save(folder/'mask'/f'{i:05}.png')
                    cond=im.copy();cond[mask]=127;Image.fromarray(cond).save(folder/'condition'/f'{i:05}.png')
                    if y is not None:Image.fromarray(y[i]).save(folder/'GT'/f'{i:05}.png')
                if y is not None:
                    diff=(comp.astype('float32')-y[i].astype('float32'))/255
                    bb=(b>0)&mask
                    def metric(region):
                        if not region.any():return None
                        mse=float((diff[region]**2).mean());return {'pixels':int(region.sum()),'MAE':float(abs(diff[region]).mean()),'PSNR':float(-10*np.log10(max(mse,1e-12)))}
                    scores.append({'frame':i,'hole':metric(mask),'protected_inside_hole':metric(bb)})
            row={'eval_id':c['eval_id'],'arm':arm,'seconds':time.time()-start,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'scores':scores,'source':c,'previous_condition':False}
            save_json(folder/f'{arm}_metrics.json',row);state['completed'].append({'eval_id':c['eval_id'],'arm':arm,'seconds':row['seconds']});state['stage']=arm;save_json(dest/'state.json',state);print(json.dumps(state['completed'][-1]),flush=True)
    state['stage']='complete';save_json(dest/'state.json',state)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan-only',action='store_true');a=p.parse_args();main(a)
