"""冻结 V4.1-R 30K 结果接入原诊断包；原生 CUDA 前向、CPU 几何，零训练更新。"""
import argparse, gc, json, os, sys, time, subprocess
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
COMMON_SOURCE = Path('/root/autodl-tmp/motion_proj_v82_gn11_v4/scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/gn11_stage1_v3_20261020/diagnostics_20261021/src')
sys.path.insert(0,str(COMMON_SOURCE))
DEFAULT_CAMPAIGN = Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v41_20261023')

def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)

def cpu_adapter():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from backend import Adapter
    return Adapter()

def binding(campaign, arm):
    from backend import digest
    run = campaign/'arms'/arm/'run30k'
    state = json.loads((run/'v41_checkpoint30000.json').read_text())
    returned = json.loads((run/'phase_0_30000_returned.json').read_text())
    if any(state.get(k) != 30000 or returned.get('phase_'+k) != 30000 for k in ('V1_updates','road_updates')):
        raise ValueError('两侧必须各完成30000更新')
    if state['arm'] != arm or state['iteration'] != 30000 or returned['status'] != 'RETURNED_AT_COMPLETE_NODE':
        raise ValueError('实验组或终态不匹配')
    paths = dict(static=run/'chkpnt30000.pth', dynamic=run/'dynamic_chkpnt30000.pth',
                 environment=run/'env_light_chkpnt30000.pth', road=run/'road_chkpnt30000.pth',
                 config=run/'effective_config.json', complete=run/'v41_checkpoint30000.json')
    records = {key:dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sha256=digest(p)) for key,p in paths.items()}
    return run, records

def unchanged(records):
    return all((Path(r['path']).stat().st_size,Path(r['path']).stat().st_mtime_ns)==(r['bytes'],r['mtime_ns']) for r in records.values())

def init_gpu(campaign, arm, adapter):
    # 不调用 Scene/readWaymoInfo，不写共享 transform，不接触训练进程。
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    sys.path.insert(0,str(campaign/'engine'))
    import torch
    from omegaconf import OmegaConf
    from scene import GaussianModel, Dynamic_GaussianModel, EnvLight
    from utils.v4_road import RoadLayer, load_bundle, POLICY as BASE_POLICY
    from utils.v41_objectives import VERSION, POLICY
    torch.set_num_threads(1)
    free,total = torch.cuda.mem_get_info()
    if free < 12*1024**3: raise RuntimeError('只读导出需要至少12GiB空闲显存；不干预其他训练')
    torch.cuda.set_per_process_memory_fraction(min(.25,8*1024**3/total))
    run = campaign/'arms'/arm/'run30k'
    config = json.loads((run/'effective_config.json').read_text())
    args = OmegaConf.create(config)
    meta,arrays = load_bundle(config['v4_bundle'])
    if not np.allclose(meta['world_to_model'],adapter.A,atol=1e-10) or meta['frame_order'] != [f'0008{r:03d}' for r in range(89,138)]:
        raise ValueError('V4和原诊断相机/坐标输入不一致')
    models = [GaussianModel(args),Dynamic_GaussianModel(args),EnvLight(resolution=args.env_map_res).cuda()]
    for prefix,model in zip(('chkpnt','dynamic_chkpnt','env_light_chkpnt'),models):
        payload,step = torch.load(str(run/f'{prefix}30000.pth'),map_location='cpu',weights_only=False,mmap=True)
        if step != 30000: raise ValueError('非30K checkpoint')
        payload = list(payload)
        if prefix == 'env_light_chkpnt': payload[0] = torch.nn.Parameter(payload[0].detach().cuda(),requires_grad=False)
        else:
            for i in range(1,11): payload[i] = payload[i].detach().cuda()
            model.time_duration = tuple(config['time_duration'])
        model.restore(tuple(payload),training_args=None)
        if getattr(model,'optimizer',None) is not None: raise RuntimeError('只读模型不应创建optimizer')
        for value in vars(model).values():
            if torch.is_tensor(value): value.requires_grad_(False)
        del payload; gc.collect()
    road = RoadLayer(arrays,float(meta['meter_to_model']),arm,'cuda')
    saved = torch.load(run/'road_chkpnt30000.pth',map_location='cpu',weights_only=False)
    if saved['version'] != VERSION or saved['policy'] != POLICY or saved['base_policy'] != BASE_POLICY or saved['dense_manifest'] != config['v41_dense_manifest'] or saved['iteration'] != 30000 or saved['arm'] != arm or saved['bundle'] != str(Path(config['v4_bundle']).resolve()):
        raise ValueError('道路sidecar绑定错误')
    frozen = {k:v.cpu().clone() for k,v in road.state_dict().items() if k in ('xyz','scaling','rotation','unit_ids')}
    road.load_state_dict(saved['layer'],strict=True)
    if not all(torch.equal(road.state_dict()[k].cpu(),v) for k,v in frozen.items()): raise ValueError('道路固定几何发生变化')
    road.requires_grad_(False)
    return run,args,models,road,arrays

def camera(adapter,raw,run):
    import torch
    from scene.cameras import Camera
    from utils.graphics_utils import getProjectionMatrixCenterShift
    c=adapter.camera(raw)
    record=next(r for r in json.loads((run/'cameras.json').read_text()) if r['img_name']==f'0008{raw:03d}')
    c2w=np.eye(4);c2w[:3,:3]=record['rotation'];c2w[:3,3]=record['position'];W=np.linalg.inv(c2w)
    if not np.allclose(W,c['W'],atol=2e-6):raise ValueError('保存相机与世界绑定不符')
    stamp=json.loads((run/'actual_scene_binding.json').read_text())['timestamps'][raw-89]
    value=Camera(raw,W[:3,:3].T.copy(),W[:3,3].copy(),cx=record['cx']/4,cy=record['cy']/4,fx=record['fx']/4,fy=record['fy']/4,
                 image=torch.zeros((3,320,480)),image_name=f'0008{raw:03d}',resolution=(480,320),timestamp=stamp)
    return value

def gpu_export(output,campaign,arm,raws):
    adapter=cpu_adapter()
    from export_inspection import measured_frame,js
    from cloud_sampling import pack
    run,records=binding(campaign,arm)
    run,args,models,road,arrays=init_gpu(campaign,arm,adapter)
    import torch
    from gaussian_renderer import prepare_rasterization
    from utils.v4_road_render import render_composite,render_road,raster
    key='v4_1_'+arm;scale=adapter.scale;checks=[];started=time.time()
    output.mkdir(parents=True,exist_ok=True)
    def cpu(t): return t.detach().cpu().numpy()
    def depth(p): return np.where(cpu(p['alpha'])>1e-8,cpu(p['depth_sum'])/np.maximum(cpu(p['alpha']),1e-8)/scale,0)[0].astype(np.float32)
    def webp(channel,raw,image):
        path=output/f'assets/{key}/{channel}/{raw:03d}.webp';path.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(np.round(np.clip(image.transpose(1,2,0),0,1)*255).astype(np.uint8)).save(path,lossless=True)
    ids=cpu(road.unit_ids);state={k:(cpu(v).astype(np.int32) if cpu(v).dtype==np.int64 else cpu(v)) for k,v in road.state_dict().items()}
    state['output_mask']=cpu(road.output_mask()).astype(np.uint8)
    state['opacity_value']=cpu(road.opacity.sigmoid())[:,0]
    js(output,f'data/{key}/road_state.js',key+'/road_state',pack(state))
    n=len(arrays['model_xyz']);record_counts=dict(created=len(ids),observed=int(cpu(road.observed).sum()),output=int(state['output_mask'].sum()))
    print('road masks '+json.dumps(record_counts),flush=True)
    with torch.inference_mode():
        for raw in raws:
            c=camera(adapter,raw,run);static,dynamic,environment=models;background=torch.tensor([1.,1.,1.] if args.white_background else [0.,0.,0.],device='cuda')
            # 三组贡献通过同一次光栅排序的one-hot feature累计，非相减/近似遮挡。
            def combined(dyn):
                from utils.v4_road_render import road_kwargs,merge_rows
                k=prepare_rasterization(c,static,args);k['means2D']=torch.zeros_like(k['means3D'])
                k['features']=torch.cat((k['features'],torch.zeros_like(k['features'][:,:1])),1)
                if dyn is not None:
                    dk=prepare_rasterization(c,dyn,args,dynamic=True);dk['means2D']=torch.zeros_like(dk['means3D'])
                    dk['features']=torch.cat((dk['features'],torch.zeros_like(dk['features'][:,:1])),1);k=merge_rows(k,dk)
                rk=road_kwargs(c,road,args,k['shs'].shape[1] if k['shs'] is not None else 1,k['colors_precomp'] is not None)
                rk['features']=torch.cat((rk['features'],torch.ones_like(rk['features'][:,:1])),1);k=merge_rows(k,rk)
                p=raster(c,args,static.active_sh_degree,k,background)
                p['raw_rgb']=p['rgb'];p['rgb']=p['rgb']+(1-p['alpha'])*environment(c.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
                return p
            joint=combined(dynamic);deleted=combined(None)
            torch.cuda.synchronize()
            independent=render_road(c,road,args)
            torch.cuda.synchronize()
            ordinary=render_composite(c,static,None,None,args,background,environment)
            native_joint=render_composite(c,static,dynamic,None,args,background,environment)
            dk=prepare_rasterization(c,dynamic,args,dynamic=True);dk['means2D']=torch.zeros_like(dk['means3D'])
            dynamic_p=raster(c,args,dynamic.active_sh_degree,dk,torch.zeros(3,device='cuda'))
            # 对照训练时保存的原V1输出；新道路不注入这个基线。
            old=np.asarray(Image.open(run/f'eval/train_30000_render/{raw-89:03d}.png')).astype(float)/255
            if old.shape==(646,966,3): old=old[2:322,484:964]
            err=float(np.abs(np.clip(cpu(native_joint['rgb']).transpose(1,2,0),0,1)-old).max())
            if err > 2/255: raise ValueError(f'原生相机/前向重放错误 raw{raw}: {err}')
            reference=render_composite(c,static,dynamic,road,args,background,environment)
            merge_error=float((reference['rgb']-joint['rgb']).abs().max())
            if merge_error>2e-5: raise ValueError('组贡献追加改变原生合成')
            # GT读取既有相同49帧冻结浮点，避免重新缩放输入。
            from backend import CAMPAIGN
            with np.load(CAMPAIGN/f'final49/full_r2/0008{raw:03d}/render_float.npz') as z:gt=z['gt'].copy();body=z['body_mask'][0].astype(np.uint8)
            measured=measured_frame(adapter,raw)
            images=dict(native_v1=cpu(native_joint['rgb']),native_static_delete=cpu(ordinary['rgb']),joint=cpu(joint['rgb']),deleted_five=cpu(deleted['rgb']),dynamic_raw=cpu(dynamic_p['rgb']),road_raw=cpu(independent['rgb']),
                        road_surface_color=cpu(independent['rgb'])/np.maximum(cpu(independent['alpha']),1e-8))
            for ch,image in images.items():webp(ch,raw,image)
            floating=dict(static_alpha=cpu(deleted['alpha'])[0],joint_alpha=cpu(joint['alpha'])[0],dynamic_alpha=cpu(dynamic_p['alpha'])[0],road_alpha=cpu(independent['alpha'])[0],
                          road_bg=cpu(deleted['extra'])[0],road_joint=cpu(joint['extra'])[0],lidar_depth_m=measured['lidar_nearest_m'],static_depth_m=depth(deleted),joint_depth_m=depth(joint),dynamic_depth_m=depth(dynamic_p),
                          body_mask=body,residual=np.abs(images['joint']-gt).mean(0))
            maps={}
            for name,value in floating.items():
                if 'depth' in name:
                    maps[name]=np.where(np.isfinite(value)&(value>0),np.round(np.clip(value*100,0,65534)),65535).astype(np.uint16)
                elif name=='body_mask':maps[name]=value
                else:maps[name]=np.round(np.clip(value,0,1)*65535).astype(np.uint16)
            js(output,f'data/{key}/{raw}.js',f'{key}/{raw}',pack(maps))
            surfaces=dict(ordinary_depth_m=depth(ordinary),ordinary_alpha=cpu(ordinary['alpha'])[0],road_depth_m=depth(independent))
            js(output,f'data/inspection/{key}_{raw}.js',f'inspection/{key}/{raw}',pack(surfaces))
            cp=np.c_[arrays['model_xyz'],np.ones(n)]@adapter.camera(raw)['W'].T;K=adapter.camera(raw)['K'];uv=cp[:,:2]/np.maximum(cp[:,2:3],1e-8)*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]-.5
            valid=(cp[:,2]>.2)&(uv[:,0]>=0)&(uv[:,0]<480)&(uv[:,1]>=0)&(uv[:,1]<320)
            x,y=np.clip(np.rint(uv).astype(int),[0,0],[479,319]).T
            units=dict(raster_uv=uv.astype(np.float32),in_view=valid.astype(np.uint8),verified_exposure=(arrays['base_weight'][raw-89]>0).astype(np.uint8),
                       road_only_alpha=floating['road_alpha'][y,x],road_alpha_in_background=floating['road_bg'][y,x],road_alpha_in_joint=floating['road_joint'][y,x],
                       road_depth_model=surfaces['road_depth_m'][y,x]*scale,source_depth_model=arrays['source_depth'][raw-89],road=images['road_surface_color'][:,y,x].T,deleted=images['deleted_five'][:,y,x].T,joint=images['joint'][:,y,x].T)
            js(output,f'data/{key}/units_{raw}.js',f'{key}/units/{raw}',pack(units))
            checks.append(dict(raw=raw,native_png_replay_max_abs=err,same_raster_group_replay_max_abs=merge_error))
            if raw in (89,100,102,113,125,137) or len(raws)==1:print(key,raw,'native PNG error',round(err,7),'seconds',round(time.time()-started,1),flush=True)
            del joint,deleted,ordinary,native_joint,independent,dynamic_p,reference,dk,c;gc.collect();torch.cuda.empty_cache()
    if not unchanged(records):raise RuntimeError('只读输入发生变化')
    receipt=dict(status='V41_NATIVE_RENDER_COMPLETE',arm=key,iteration=30000,frames=list(raws),checkpoints=records,checks=checks,road=record_counts,
                 created_utc=datetime.now(timezone.utc).isoformat(),gpu_used=True,optimizer_created=False,optimizer_updates=0,training_updates=0,
                 geometry_fixed_verified=True,protected_inputs_unchanged=True,render_seconds=round(time.time()-started,2),quality_accepted=False,human_verdict=None,
                 cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                 inference_memory_fraction=min(.25,8*1024**3/torch.cuda.get_device_properties(0).total_memory),
                 concurrent_training_note='B keeps training independently; device-wide NVML includes this read-only inference interval')
    save_json(output/f'data/{key}/render_receipt.json',receipt)
    return receipt

def geometry_export(output,campaign,arm):
    adapter=cpu_adapter()
    import torch
    from export_inspection import js
    from cloud_sampling import Sampler,pack,hashes
    from export_offline import attributes,memberships
    run,records=binding(campaign,arm);key='v4_1_'+arm;branches=[]
    for name,prefix in [('static','chkpnt'),('dynamic','dynamic_chkpnt')]:
        payload,step=torch.load(str(run/f'{prefix}30000.pth'),map_location='cpu',weights_only=False,mmap=True)
        xyz,dc,rest,logs,q,logit,t,logt,vel=[payload[i].detach().numpy() for i in range(1,10)]
        branches.append(dict(name=name,sha=records[name]['sha256'],xyz=xyz,dc=dc,rest=rest,scale=np.exp(logs),q=q,
                             opacity=1/(1+np.exp(-np.clip(logit[:,0],-80,80))),t=t[:,0],sigma_t=np.exp(logt[:,0]),vel=vel,cycle=float(payload[16]),road=np.full(len(xyz),-1,np.int64)))
        del payload;gc.collect()
    saved=torch.load(run/'road_chkpnt30000.pth',map_location='cpu',weights_only=False)['layer'];s={k:v.detach().numpy() for k,v in saved.items()}
    keep=((s['observed']&s['source_seen'])|(s['prior']&s['appearance_updated']))&(s['steps']>0);n=int(keep.sum());static=branches[0]
    add=dict(xyz=s['xyz'][keep],dc=s['dc'][keep],rest=np.zeros((n,static['rest'].shape[1],3),np.float32),scale=s['scaling'][keep],q=s['rotation'][keep],
             opacity=1/(1+np.exp(-s['opacity'][keep,0])),t=np.zeros(n,np.float32),sigma_t=np.full(n,1e10,np.float32),vel=np.zeros((n,3),np.float32),road=s['unit_ids'][keep])
    for k,v in add.items():static[k]=np.concatenate((static[k],v))
    # 拼接row命名只对当前导出有效；实际来源静态/道路各自SHA随元数据保留。
    static['sha']=records['static']['sha256']
    sampler=Sampler(adapter);sampler.current=key;sampler.model=dict(arm=key,branches=branches)
    original_get=sampler.get_model;sampler.get_model=lambda requested:sampler.model if requested==key else original_get(requested)
    cloud=sampler.query(dict(arm=key,raw=102));cloud['road_checkpoint_sha256']=records['road']['sha256'];cloud['road_output_only']=True
    js(output,f'data/{key}/cloud.js',key+'/cloud',cloud)
    meta=dict(branches={},members={},scopes=['projection']);orders={};size=32768
    for b in branches:
        ordinary=np.flatnonzero(b['road']<0);order=ordinary[np.lexsort((ordinary,hashes(ordinary)))];orders[b['name']]=order
        rec=dict(total=len(b['xyz']),ordinary_total=len(order),road_total=int((b['road']>=0).sum()),checkpoint_sha256=b['sha'],chunks=[])
        for i,start in enumerate(range(0,len(order),size)):
            ids=order[start:start+size];path=f'data/geometry/{key}/{b["name"]}_{i:03d}.js';chunk_key=f'geometry/{key}/{b["name"]}/{i}'
            js(output,path,chunk_key,pack(attributes(b,ids)));rec['chunks'].append(dict(path=path,key=chunk_key,start=start,count=len(ids)))
        meta['branches'][b['name']]=rec
    for raw in range(89,138):
        arrays={b['name']+'_projection':memberships(sampler,b,raw,orders[b['name']],include_box=False)[0] for b in branches}
        path=f'data/geometry/{key}/members_{raw}.js';chunk_key=f'geometry/{key}/members/{raw}';js(output,path,chunk_key,pack(arrays));meta['members'][str(raw)]=dict(path=path,key=chunk_key)
    js(output,f'data/{key}/geometry_index.js',key+'/geometry_index',meta)
    if not unchanged(records):raise RuntimeError('几何导出期间原始输入发生变化')
    print('geometry ready '+key,flush=True)
    return meta

def finish(output,campaign,arm):
    from backend import digest
    key='v4_1_'+arm;receipt=json.loads((output/f'data/{key}/render_receipt.json').read_text())
    if receipt['frames']!=list(range(89,138)):raise ValueError('必须完整49帧后才发布可用')
    required=[output/f'data/{key}/{name}' for name in ('cloud.js','road_state.js','geometry_index.js')]
    for raw in range(89,138):
        required += [output/f'data/{key}/{raw}.js',output/f'data/{key}/units_{raw}.js',output/f'data/inspection/{key}_{raw}.js',output/f'data/geometry/{key}/members_{raw}.js']
        required += [output/f'assets/{key}/{channel}/{raw:03d}.webp' for channel in ('joint','deleted_five','dynamic_raw','road_raw','road_surface_color','native_v1','native_static_delete')]
    if any(not p.is_file() or p.stat().st_size<50 for p in required):raise ValueError('V4诊断文件不完整，保留占位')
    files=[]
    for folder in (output/f'data/{key}',output/f'assets/{key}',output/f'data/geometry/{key}'):
        files.extend(p for p in folder.rglob('*') if p.is_file() and p.name!='ready.json')
    files.extend(output/f'data/inspection/{key}_{raw}.js' for raw in range(89,138))
    manifest={str(p.relative_to(output)).replace('\\','/'):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in files}
    receipt.update(status='V41_DIAGNOSTICS_READY',asset_files=manifest,geometry_complete=True,source_optimization_log_available=False)
    from validate_v41_results import validate
    receipt['validation']=validate(output,arm,receipt)
    save_json(output/f'data/{key}/ready.json',receipt)
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--campaign',type=Path,default=DEFAULT_CAMPAIGN);p.add_argument('--arms',nargs='+',choices=['R'],default=['R']);p.add_argument('--raws',nargs='+',type=int,default=list(range(89,138)));p.add_argument('--phase',choices=['render','geometry','finish','all'],default='all');a=p.parse_args()
    if a.phase == 'all':
        for phase in ('render','geometry','finish'):
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--output',str(a.output),'--campaign',str(a.campaign),'--arms',*a.arms,'--phase',phase,'--raws',*map(str,a.raws)],check=True)
        return
    for arm in a.arms:
        if a.phase in ('render','all'):gpu_export(a.output,a.campaign,arm,a.raws)
        if a.phase in ('geometry','all'):geometry_export(a.output,a.campaign,arm)
        if a.phase in ('finish','all'):finish(a.output,a.campaign,arm)

if __name__=='__main__':main()
