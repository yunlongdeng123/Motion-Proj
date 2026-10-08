"""同一主干checkpoint分别接官方和固定r47；正常10帧窗口，f05图像审核。"""
from pathlib import Path
import argparse,json,os,sys,time
import numpy as np
from PIL import Image
import torch

REPO=Path('/root/autodl-tmp/motion_proj_v77')
S=REPO/'scripts/worldsim_v77/target_protected'
sys.path.insert(0,str(S/'iteration14'))
import common
from interface import DeletionRequest,encode_references
from multi_prior import MultiPriorBranch
from engine import MultiPriorDeletionEngine
from repair_drive import set_seed
from safetensors.torch import load_file
sys.path.insert(0,str(Path(__file__).parent))
from finetune_scope import selected

T=common.T;ROOT=T/'r52';OLD=T/'r51/r47_comparison/inputs/R001'

def main(a):
    size=tuple(a.size)
    if any(d<=0 or d%8 for d in size):raise ValueError('推理尺寸必须为正且可被8整除')
    if not a.proxy_only and size!=(576,1024):raise ValueError('R001几何条件仅支持576x1024；同尺度诊断请用--proxy-only')
    if a.proxy_only and a.frames!=10:raise ValueError('同尺度proxy容量诊断必须使用真实10帧窗口')
    if a.static_source_frame is not None and (a.proxy_only or a.frames!=10 or a.proxy):
        raise ValueError('静态容量窗仅用于R001两臂，10重复帧，不混入proxy')
    out=ROOT/'evaluation'/a.label;out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise RuntimeError('已有完整结果，禁止覆盖')
    import fcntl
    lock=open(ROOT/'gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4)
    branch=MultiPriorBranch().cuda()
    branch.load_state_dict(load_file(str(T/'r47/training/branch_0320.safetensors')),strict=True)
    branch.requires_grad_(False)
    engine=MultiPriorDeletionEngine(branch,num_frames=a.frames,out_size=size,num_steps=25)
    if a.weights:
        weights=load_file(str(a.weights));params=dict(engine.model.named_parameters())
        required={k for k in params if selected(k)}
        assert set(weights)==required,'微调checkpoint范围不一致'
        with torch.no_grad():
            for k,v in weights.items():params[k].copy_(v.to(device=params[k].device,dtype=params[k].dtype))
        del weights
    capacity_metadata={}
    if a.proxy_only:
        # 直接复用训练的先遮后缩合同；仅官方主路，参考和先验均为空占位。
        from train_one import load_pair
        capacity_metadata,(yy,cc,hh),_=load_pair(a.capacity_pair or ROOT/'pairs/R001_realRGB_clip_v2/pair.json',size)
        known_y=np.rint((yy.permute(0,2,3,1).numpy()+1)*127.5).clip(0,255).astype('uint8')
        masked_rgb=np.rint((cc.permute(0,2,3,1).numpy()+1)*127.5).clip(0,255).astype('uint8')
        hole=hh.numpy()
        priors={'geometry':np.zeros((a.frames,12,size[0]//8,size[1]//8),dtype='float32'),
                'references':np.full((1,1,1,3),127,dtype='uint8'),
                'reference_valid':np.zeros((1,1,1),dtype=bool)}
        req=DeletionRequest(masked_rgb,hole,hole.astype('float32'),priors,{'references':[]})
        jobs=[('proxy_official',req,'baseline',cc)]
        ref=None
    else:
        priors=dict(np.load(OLD/'condition.npz'))
        if a.frames==1:
            for key in ('geometry','bev','bev_rays','query_time','parameters','hole'):priors[key]=priors[key][5:6].copy()
        elif a.static_source_frame is not None:
            for key in ('geometry','bev','bev_rays','query_time','parameters','hole'):
                f=a.static_source_frame;priors[key]=np.repeat(priors[key][f:f+1],10,axis=0)
        refs=json.loads((OLD/'references.json').read_text())
        source=T/'r50/inputs/R001'
        def stack(folder,role,indices):
            paths=sorted(folder.glob('*.jpg' if role=='rgb' else '*.png'))
            return np.stack([np.asarray(Image.open(paths[i]).convert('RGB' if role=='rgb' else 'L')) for i in indices])
        indices=[5] if a.frames==1 else ([a.static_source_frame]*10 if a.static_source_frame is not None else list(range(10)))
        rgb=stack(source/'rgb','rgb',indices)
        hole=stack(source/'model_mask','mask',indices)>0
        alpha=stack(source/'alpha','mask',indices).astype('float32')/255
        req=DeletionRequest(rgb,hole,alpha,priors,refs)
        ref=encode_references(engine.model,req)
        jobs=[('R001_official',req,'baseline',None),('R001_r47',req,'RGB_and_geometry',None)]
        if a.proxy:
            pair=ROOT/'pairs/R001_realRGB_clip_v2'
            px=np.stack([np.asarray(Image.open(pair/'x'/f'{i:05}.png').convert('RGB')) for i in indices])
            ph=np.stack([np.asarray(Image.open(pair/'hole'/f'{i:05}.png').convert('L')) for i in indices])>0
            jobs.append(('proxy_official',DeletionRequest(px,ph,ph.astype('float32'),priors,refs),'baseline',None))
    results=[]
    for name,request,arm,masked_condition in jobs:
        dest=out/name;dest.mkdir(exist_ok=True)
        if not a.weights and a.frames==10 and a.static_source_frame is None and name.startswith('R001_'):
            import shutil
            old=(T/'r50/inputs/R001') if name=='R001_official' else (T/'r51/r47_comparison/evaluation/R001')
            for role,folder in [('native','native'),('compose','compose')]:
                files=sorted((old/folder).glob('*.png'));assert len(files)==10,(old,folder,len(files))
                shutil.copytree(old/folder,dest/(role+'_frames'),dirs_exist_ok=True)
                shutil.copy2(files[5],dest/f'{role}.png')
            results.append({'name':name,'reused_from':str(old),'num_frames':10,'resolution':list(size),'new_GPU_window':False})
            continue
        engine.get_deletion(request,arm=arm,reference_latents=ref)
        if masked_condition is not None:
            engine.masked_condition=masked_condition
        set_seed(42);torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        with torch.inference_mode():engine.predict(1,False,'Deletion')
        raw=np.stack(engine.im_result);assert raw.shape==(a.frames,*size,3)
        comp=np.where(request.edit_mask[...,None],raw,known_y) if a.proxy_only else request.compose(raw)
        for role,values in [('native',raw),('compose',comp)]:
            (dest/(role+'_frames')).mkdir(exist_ok=True)
            for i,val in enumerate(values):Image.fromarray(val).save(dest/(role+'_frames')/f'{i:05}.png')
            Image.fromarray(values[0 if a.frames==1 else 5]).save(dest/f'{role}.png')
        r={'name':name,'arm':arm,'seconds':time.monotonic()-started,'peak_GiB':torch.cuda.max_memory_allocated()/2**30,
           'weights':str(a.weights) if a.weights else 'official',
           'r47_branch':'disabled' if arm=='baseline' else 'fixed step320',
           'num_frames':a.frames,'resolution':list(size),'seed':42,'steps':25,'query_frame':5,
           'static_repeat_capacity_only':a.static_source_frame is not None or capacity_metadata.get('static_repeat_capacity_only',False),
           'source_frame_index':a.static_source_frame if a.static_source_frame is not None else capacity_metadata.get('source_frame_index',5),
           'capacity_pair':str(a.capacity_pair) if a.capacity_pair else None,
           'input_contract':'train.resized clear-before-resize' if masked_condition is not None else 'native full resolution',
           'CFG_prior_calls':engine.prior_cfg_calls.copy(),
           'human_verdict':None,'temporal_verdict':None}
        if a.proxy_only:
            error=np.abs(raw.astype('float32')-known_y.astype('float32'))
            r['known_Y_hole_MAE']=float(error[request.edit_mask].mean())
            r['known_Y_hole_MAE_by_frame']=[float(error[i][request.edit_mask[i]].mean()) for i in range(a.frames)]
        results.append(r);print(json.dumps(r),flush=True)
    (out/'result.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--weights',type=Path);p.add_argument('--label',required=True)
    p.add_argument('--proxy',action='store_true');p.add_argument('--proxy-only',action='store_true')
    p.add_argument('--size',type=int,nargs=2,default=[576,1024]);p.add_argument('--frames',type=int,choices=[1,10],default=10)
    p.add_argument('--static-source-frame',type=int,choices=range(10))
    p.add_argument('--capacity-pair',type=Path)
    main(p.parse_args())
