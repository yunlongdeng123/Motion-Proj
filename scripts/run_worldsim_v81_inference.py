"""官方 backbone 冻结推理入口；默认仅输出任务，不自动占用 GPU。"""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path('/root/autodl-tmp')
REPOS={'dvgt':ROOT/'external/worldsim_v81/DVGT','vggt':ROOT/'external/worldsim_v72/vggt','dggt':ROOT/'external/worldsim_v81/dggt'}
WEIGHTS={'dvgt':ROOT/'models/worldsim_v81/dvgt1.pt','vggt':ROOT/'models/eas_vggt/vggt/model.safetensors','dggt':ROOT/'models/worldsim_v81/model_latest_nuscenes.pt'}

def select_views(manifest,variant,anchor_camera=None):
    views=manifest['views']
    if anchor_camera:
        offset=next(i for i,v in enumerate(views) if v['camera']==anchor_camera)
        views=views[offset:]+views[:offset]
    if variant=='full6':return views
    if variant=='sparse3':return [v for i,v in enumerate(views) if i in [0,2,4]]
    if variant=='sparse2':return [views[0],views[3]]
    if variant=='temporal18':
        by_sample={}
        for v in manifest['context_views']:by_sample.setdefault(v['sample_token'],[]).append(v)
        t=views[0]['timestamp_us'];prev=[v for v in by_sample.values() if v[0]['timestamp_us']<t];nxt=[v for v in by_sample.values() if v[0]['timestamp_us']>t]
        if not prev or not nxt:raise ValueError('temporal18 requires both adjacent complete frames')
        before=max(prev,key=lambda v:v[0]['timestamp_us']);after=min(nxt,key=lambda v:v[0]['timestamp_us'])
        for group in [before,after]:
            if abs(abs(group[0]['timestamp_us']-t)-500000)>120000:raise ValueError('DVGT requires 2Hz; missing adjacent frame')
        return before+views+after
    raise ValueError(variant)

def pixel_affine(size,model):
    width,height=size;tw=512 if model=='dvgt' else 518;multiple=16 if model=='dvgt' else 14
    th=round(height*(tw/width)/multiple)*multiple;crop=max(0,(th-tw)//2)
    return [tw, min(th,tw)], [[tw/width,0,(tw/width-1)/2],[0,th/height,(th/height-1)/2-crop],[0,0,1]]

def scale_from_cameras(extrinsics,views):
    # 使用公开相机基线确定尺度，不接触 LiDAR 或被评 ROI。
    centers=np.array([-E[:,:3].T@E[:,3] for E in extrinsics])
    true=np.array([np.array(v['world_from_camera'])[:3,3] for v in views])
    ratios=[]
    for i in range(len(views)):
        for j in range(i):
            d=np.linalg.norm(centers[i]-centers[j]);g=np.linalg.norm(true[i]-true[j])
            if d>.01 and g>.1:ratios.append(g/d)
    if not ratios:raise ValueError('metric scale is unidentified from available camera baselines')
    return float(np.median(ratios)),float(np.std(ratios)/np.mean(ratios))

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=REPOS,required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--variant',choices=['full6','sparse3','sparse2','temporal18'],default='full6');p.add_argument('--out',required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--texture-case',help='JSON containing per-view intervention images; synthetic diagnostics only')
    p.add_argument('--anchor-camera',help='Keep the evaluated ROI camera in all view subsets')
    a=p.parse_args();m=json.loads(Path(a.manifest).read_text());views=select_views(m,a.variant,a.anchor_camera)
    if a.texture_case:
        tx=json.loads(Path(a.texture_case).read_text());overrides=tx['image_overrides']
        views=[dict(v,image=overrides.get(v['image'],v['image'])) for v in views]
    plan={'method':a.method,'variant':a.variant,'window':m['window_id'],'n_images':len(views),'anchor_camera':a.anchor_camera,'checkpoint':str(WEIGHTS[a.method]),'role':'SYNTHETIC_DIAGNOSTIC' if a.texture_case else 'DISCOVERY','heldout_access':False}
    if not a.execute:print(json.dumps(plan,indent=2));return
    import torch
    if not torch.cuda.is_available():raise RuntimeError('WAIT_GPU: no GPU; CPU inference is intentionally disabled')
    torch.set_num_threads(1);torch.manual_seed(8101);out=Path(a.out)
    if (out/'result.json').exists():raise FileExistsError('completed run exists; select a new output run')
    out.mkdir(parents=True,exist_ok=True);repo=REPOS[a.method];sys.path.insert(0,str(repo));os.chdir(repo)
    if a.method=='dvgt':
        from dvgt.models.architectures.dvgt1 import DVGT1
        from dvgt.utils.load_fn import load_and_preprocess_images
        # 官方完整权重含 DINO；只禁用构造阶段冗余下载，随后 strict=True 覆盖全部参数。
        hubload=torch.hub.load
        def local_hub(*args,**kwargs):
            if args and 'dinov3' in str(args[0]):kwargs['pretrained']=False;kwargs.pop('weights',None)
            return hubload(*args,**kwargs)
        torch.hub.load=local_hub
        try:model=DVGT1(dino_v3_weight_path=None,frames_chunk_size=1)
        finally:torch.hub.load=hubload
        state=torch.load(WEIGHTS[a.method],map_location='cpu',weights_only=True,mmap=True)
        model.load_state_dict(state,strict=True);del state
        temp=out/'native_input';temp.mkdir(exist_ok=True);sample_order=list(dict.fromkeys(v['sample_token'] for v in views))
        for v in views:
            fi=sample_order.index(v['sample_token']);d=temp/f'frame_{fi}';d.mkdir(exist_ok=True)
            target=d/(str([u['camera'] for u in views if u['sample_token']==v['sample_token']].index(v['camera'])).zfill(2)+'_'+v['camera']+'.jpg')
            if not target.exists():target.symlink_to(v['image'])
        images=load_and_preprocess_images(str(temp),mode='crop').to('cuda')
    else:
        if a.method=='vggt':
            from vggt.models.vggt import VGGT
            from vggt.utils.load_fn import load_and_preprocess_images
            from safetensors.torch import load_file
            model=VGGT();state=load_file(str(WEIGHTS[a.method]))
        else:
            from dggt.models.vggt import VGGT
            # 使用同一官方预处理函数；DGGT dataset 文件还导入其原生包。
            from datasets.dataset import load_and_preprocess_images
            model=VGGT();state=torch.load(WEIGHTS[a.method],map_location='cpu',weights_only=True,mmap=True)
        model.load_state_dict(state,strict=True);del state
        images=load_and_preprocess_images([v['image'] for v in views],mode='crop').to('cuda')
    model=model.eval().to('cuda');torch.cuda.reset_peak_memory_stats();started=time.time()
    dtype=torch.bfloat16 if torch.cuda.get_device_capability()[0]>=8 else torch.float16
    with torch.inference_mode(),torch.autocast('cuda',dtype=dtype):pred=model(images)
    torch.cuda.synchronize();runtime=time.time()-started
    # 保存原生 tensor，不按置信度丢弃难点。renderer 单独消费 DGGT gs_map。
    keys=['points','points_conf','absolute_ego_pose_enc'] if a.method=='dvgt' else ['depth','depth_conf','world_points','world_points_conf','pose_enc','gs_map','gs_conf','dynamic_conf','semantic_logits']
    saved={k:pred[k].detach().float().cpu().numpy() for k in keys if k in pred}
    np.savez_compressed(out/'native_outputs.npz',**saved)
    scale=1.;scale_cv=None
    if a.method!='dvgt':
        if a.method=='vggt':from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        else:from dggt.utils.pose_enc import pose_encoding_to_extri_intri
        ext,intr=pose_encoding_to_extri_intri(pred['pose_enc'],images.shape[-2:]);ext=ext[0].float().cpu().numpy()
        scale,scale_cv=scale_from_cameras(ext,views)
        np.savez_compressed(out/'cameras.npz',extrinsic=ext,intrinsic=intr[0].float().cpu().numpy())
    records=[]
    for i,v in enumerate(views):
        if v['sample_token']!=m['target_sample']:continue
        size=list(Image.open(v['image']).size);net_size,A=pixel_affine(size,a.method)
        if a.method=='dvgt':
            # DVGT points 位于第一输入时刻 ego frame。该变换来自输入 calibration contract。
            if 'world_from_ego' not in views[0]:raise ValueError('input manifest lacks first-frame world_from_ego')
            points=saved['points'][0].reshape(-1,*saved['points'].shape[-3:])[i]
            T=np.linalg.inv(np.array(v['world_from_camera']))@np.array(views[0]['world_from_ego'])
            xyz=points@T[:3,:3].T+T[:3,3];depth=xyz[...,2]
        else:depth=saved['depth'][0,i,...,0]*scale
        np.save(out/(v['camera']+'_depth_z_m.npy'),depth.astype('float32'))
        records.append({'camera':v['camera'],'original_wh':size,'network_wh':net_size,'original_to_network_pixel_center':A})
    result={**plan,'status':'DONE','elapsed_s':runtime,'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'metric_scale':scale,'camera_baseline_scale_cv':scale_cv,'scale_source':'native_metric' if a.method=='dvgt' else 'dataset_camera_baselines_no_lidar', 'views':records,'render_status':'NOT_RENDERED','failure_codes':[]}
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
