"""Read-only native motion/camera diagnostic baseline, no new training method.

Single physical dynamic target avoids inventing row-level ownership. Gaussian
means AND covariance rotations receive the requested rigid pose. SH appearance
is inherited in world coordinates; novel sides/lighting remain a known limit.
"""
import argparse,copy,gc,importlib.util,json,os,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation
from resource_control import require_start,stop_requested
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
SOURCE=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/training_source/scene0008_089_137')
HELPER=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v41_20261023/engine/utils/v4_road_render.py')
sys.path.insert(0,str(SOURCE))
import torch
from omegaconf import OmegaConf
from scene import GaussianModel,Dynamic_GaussianModel,EnvLight
from scene.cameras import Camera
from gaussian_renderer import prepare_rasterization,render as actual_native_render
spec=importlib.util.spec_from_file_location('frozen_raster_helpers',HELPER);helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)

def read(p):return json.loads(Path(p).read_text())
def cpu(t):return t.detach().cpu().numpy()
def save(p,t):Image.fromarray(np.round(np.clip(cpu(t).transpose(1,2,0),0,1)*255).astype('uint8')).save(p)
def quat_product(a,b):
    aw,ax,ay,az=a.unbind(-1);bw,bx,by,bz=b.unbind(-1)
    q=torch.stack((aw*bw-ax*bx-ay*by-az*bz,aw*bx+ax*bw+ay*bz-az*by,aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw),-1)
    return q/q.norm(dim=-1,keepdim=True).clamp_min(1e-12)

class NativeMotion:
    def __init__(self,run,iteration):
        self.run=run;self.config=read(run/'effective_config.json');self.args=OmegaConf.create(self.config)
        self.source=Path(self.config['source_path']);self.iteration=iteration
        self.sh_frame='world'
        assert not self.args.compute_cov3D_python and not self.args.convert_SHs_python,'Frozen baseline uses quaternion covariance and SH raster colors'
        assert len(read(self.source/'dynamic_multi_actor_preparation.json')['actors'])==1,'Do not pretend arbitrary multi-actor rows are identified'
        event=read(run/'checkpoint_events'/f'iteration_{iteration:06d}.json');assert event['status']=='complete' and len(event['frames'])==49
        self.files=[run/f'{p}{iteration}.pth' for p in ['chkpnt','dynamic_chkpnt','env_light_chkpnt','training_state']]
        self.before={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in self.files}
        self.static=GaussianModel(self.args);self.dynamic=Dynamic_GaussianModel(self.args);self.env=EnvLight(resolution=self.args.env_map_res).cuda()
        for p,model in zip(self.files,[self.static,self.dynamic,self.env]):
            payload,step=torch.load(str(p),map_location='cpu',mmap=True);assert step==iteration;payload=list(payload)
            if 'env_light' in p.name:payload[0]=torch.nn.Parameter(payload[0].detach().cuda(),requires_grad=False)
            else:
                for i in range(1,11):payload[i]=payload[i].detach().cuda()
                model.time_duration=tuple(self.config['time_duration'])
            model.restore(tuple(payload),training_args=None);assert getattr(model,'optimizer',None) is None
            for value in vars(model).values():
                if torch.is_tensor(value):value.requires_grad_(False)
            del payload;gc.collect()
        self.records=read(run/'cameras.json');self.stamps=read(run/'actual_scene_binding.json')['timestamps'];assert len(self.records)==len(self.stamps)==49
        self.bg=torch.tensor([1.,1.,1.] if self.args.white_background else [0.,0.,0.],device='cuda')
        self.T=np.loadtxt(self.source/'transform_matrix.txt');self.Tinv=np.linalg.inv(self.T)
        self.scale=float(np.linalg.norm(self.T[:3,0]));assert np.allclose(self.T[:3,:3].T@self.T[:3,:3],np.eye(3)*self.scale**2,atol=1e-9)
        self.versions=self.tensor_versions()

    def tensor_versions(self):return {(name,key):(v._version,tuple(v.shape)) for name,m in [('static',self.static),('dynamic',self.dynamic),('env',self.env)] for key,v in vars(m).items() if torch.is_tensor(v)}
    def unchanged(self):return self.versions==self.tensor_versions() and self.before=={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in self.files}

    def camera(self,row,case,query=False):
        i=row['camera_index'];r=self.records[i];assert r['img_name']==row['frame_id']
        c2w=np.eye(4);c2w[:3,:3]=r['rotation'];c2w[:3,3]=r['position']
        if case.get('camera_lateral_m'):c2w[:3,3]+=c2w[:3,0]*self.scale*case['camera_lateral_m']
        if case.get('camera_yaw_deg'):
            from prepare_motion_suite import ry
            c2w[:3,:3]=c2w[:3,:3]@ry(np.deg2rad(case['camera_yaw_deg']))
        w=np.linalg.inv(c2w);timestamp=row['dynamic_query_model_time'] if query else row['camera_model_time']
        return Camera(i,w[:3,:3].T.copy(),w[:3,3].copy(),cx=r['cx']/4,cy=r['cy']/4,fx=r['fx']/4,fy=r['fy']/4,image=torch.zeros(3,320,480),image_name=r['img_name'],resolution=(480,320),timestamp=timestamp)

    def dynamic_kwargs(self,camera,row,case,edited):
        value=prepare_rasterization(camera,self.dynamic,self.args,dynamic=True);value['means2D']=torch.zeros_like(value['means3D'])
        if edited and case['family'] in ('translate','insert_reused','turn'):
            world=np.eye(4);world[:3,:3]=row['target_rotation_delta_world'];world[:3,3]=row['target_affine_translation_world']
            model=self.T@world@self.Tinv;R=model[:3,:3];offset=model[:3,3]
            value['means3D']=value['means3D']@torch.tensor(R.T,dtype=torch.float32,device='cuda')+torch.tensor(offset,dtype=torch.float32,device='cuda')
            if case['family']=='turn':
                q=Rotation.from_matrix(R).as_quat();q=torch.tensor(np.roll(q,1),dtype=torch.float32,device='cuda')
                value['rotations']=quat_product(q,value['rotations'])
                if self.sh_frame=='object':
                    from sh_rotation_control import rotate
                    value['shs']=rotate(value['shs'],R)
        return value

    def render(self,row,case):
        camera=self.camera(row,case);static=prepare_rasterization(camera,self.static,self.args);static['means2D']=torch.zeros_like(static['means3D'])
        static['features']=torch.cat((static['features'],torch.zeros_like(static['features'][:,:1])),1);kwargs=static
        def add(dynamic):
            nonlocal kwargs
            z=torch.cat((dynamic['means3D'],torch.ones_like(dynamic['means3D'][:,:1])),1)@camera.world_view_transform[:,2:3]
            dynamic['features']=torch.cat((z,torch.ones_like(z),torch.ones_like(z)),1)
            kwargs=helpers.merge_rows(kwargs,dynamic)
        if case['family']!='delete':
            query=self.camera(row,case,query=True)
            if case['family']=='insert_reused':
                add(self.dynamic_kwargs(camera,row,case,False))
                add(self.dynamic_kwargs(query,row,case,True))
            else:add(self.dynamic_kwargs(query,row,case,True))
        out=helpers.raster(camera,self.args,self.static.active_sh_degree,kwargs,self.bg)
        env=self.env(camera.get_world_directions(False).permute(1,2,0)).permute(2,0,1);out['rgb']=out['rgb']+(1-out['alpha'])*env
        return out

def continuous_revisions(n,specs,suite,destination):
    """Persistent request-state baseline, not a learned world-memory method.

    The original is removed once. Later revisions manipulate the same clone,
    rather than implicitly re-inserting the original or accumulating clones.
    """
    by_id={x['id']:x for x in specs};protocol=suite['continuous_sequence'];states={'native':dict(original_visible=True,clone=None,camera_lateral_m=0.)};results=[]
    sensor=read(Path(suite['source_sensor'])/'sensor_preparation.json')
    anchor=np.loadtxt(n.source/'pose'/(sensor['frames'][0]['frame_id']+'.txt'))
    world_left=anchor[:3,1]
    for request in protocol['requests']:
        state=copy.deepcopy(states[request['parent']]);op=request['operation']
        if op=='delete':state['original_visible']=False
        elif op=='insert_reused':
            assert state['clone'] is None and not state['original_visible']
            state['clone']=dict(instance_id=request['new_instance'],absolute_lateral_m=request['lateral_m'],yaw_deg=0.)
        elif op=='translate':
            assert state['clone']['instance_id']==request['target'];state['clone']['absolute_lateral_m']=request['absolute_lateral_m']
        elif op=='turn':
            assert state['clone']['instance_id']==request['target'];state['clone']['yaw_deg']=request['final_yaw_deg']
        elif op=='camera_lateral':state['camera_lateral_m']=request['camera_lateral_m']
        elif op=='camera_native':state['camera_lateral_m']=0.
        else:raise ValueError(op)
        states[request['revision']]=state
    revisit_max=0.;cache_control_max=0.;cache_hits=0;cached_states={}
    for request in protocol['requests']:
        state=states[request['revision']];directory=destination/request['revision']
        for sub in ['native_rgb','float']:(directory/sub).mkdir(parents=True)
        case=dict(family='delete',camera_lateral_m=state['camera_lateral_m']);rows=copy.deepcopy(by_id['delete_target']['frames'])
        if state['clone'] is not None:
            clone=state['clone'];assert not state['original_visible']
            if clone['yaw_deg']:
                assert clone['yaw_deg']==15.;rows=copy.deepcopy(by_id['turn_left15']['frames']);case['family']='turn'
                for row in rows:row['target_affine_translation_world']=(np.array(row['target_affine_translation_world'])+world_left*clone['absolute_lateral_m']).tolist()
            else:
                # Current protocol uses absolute lateral2m then1m. One visible
                # asset is rendered, preserving the deleted-original state.
                name='insert_reused_lateral_2m' if clone['absolute_lateral_m']==2. else 'translate_lateral_1m'
                assert clone['absolute_lateral_m'] in (1.,2.);rows=copy.deepcopy(by_id[name]['frames']);case['family']='translate'
        state_key=json.dumps(state,sort_keys=True);cached_directory=cached_states.get(state_key);records=[]
        for row in rows:
            require_start();out=n.render(row,case);name=f'{row["raw"]:03d}';save(directory/'native_rgb'/(name+'.png'),out['rgb'])
            np.savez_compressed(directory/'float'/(name+'.npz'),rgb=cpu(out['rgb']),alpha=cpu(out['alpha']),depth_sum=cpu(out['depth_sum']),dynamic_weight=cpu(out['extra']))
            # Plain deterministic cache is a strong engineering control. It
            # cannot establish a benefit from learned persistent memory.
            if cached_directory is not None:
                cached=np.load(cached_directory/'float'/(name+'.npz'))['rgb'];cache_control_max=max(cache_control_max,float(np.abs(cached-cpu(out['rgb'])).max()));cache_hits+=1
            if request['revision']=='revisit_original_camera':
                reference=np.load(destination/'turn_inserted/float'/(name+'.npz'))['rgb'];revisit_max=max(revisit_max,float(np.abs(cpu(out['rgb'])-reference).max()))
            records.append(dict(raw=row['raw'],active_instances=[] if state['clone'] is None else [state['clone']['instance_id']],original_visible=state['original_visible']))
        results.append(dict(revision=request['revision'],parent=request['parent'],state=state,frames=records))
        cached_states.setdefault(state_key,directory)
    assert revisit_max==0 and cache_control_max==0 and n.unchanged()
    result=dict(status='native_continuous_revision_baseline_complete',revisions=results,revision_count=len(results),frames_per_revision=31,original_checkpoint_updated=False,plain_cache_control_max_difference=cache_control_max,plain_state_cache_hits=cache_hits,revisit_same_revision_original_camera_max_difference=revisit_max,new_training_updates=0,backend_executed=False,persistent_memory_algorithm_gain_established=False,quality_accepted=False,human_verdict=None,limitations=['one reused clone, not an independently reconstructed second asset','identity/revisit equalities are engineering checks, not learned repair or memory gains','fixed video backend for continuous requests remains pending'])
    (destination/'summary.json').write_text(json.dumps(result,indent=2));return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--suite',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--iteration',type=int,default=30000);p.add_argument('--single-frame',action='store_true');p.add_argument('--sh-frame',choices=['world','object'],default='world');p.add_argument('--only-turn',action='store_true');a=p.parse_args()
    require_start();assert not a.output.exists(),'No-clobber, preserve completed or failed diagnostic attempt';a.output.mkdir(parents=True)
    torch.set_num_threads(1);started=time.monotonic();n=NativeMotion(a.run,a.iteration);n.sh_frame=a.sh_frame;suite=read(a.suite/'summary.json')
    specs=[read(a.suite/(x['id']+'.json')) for x in suite['prepared_single_requests']]
    if a.only_turn:specs=[x for x in specs if x['family'] in ('no_edit','turn')]
    base=next(x for x in specs if x['family']=='no_edit');frame_ids=[15] if a.single_frame else range(31)
    for case in specs:
        for sub in ['native_rgb','float','mask']:(a.output/case['id']/sub).mkdir(parents=True)
    records=[];maximum_identity=0.;baseline_first=None
    with torch.no_grad():
        for i in frame_ids:
            require_start();row=base['frames'][i];camera=n.camera(row,base);baseline=n.render(row,base)
            actual=actual_native_render(camera,n.static,n.dynamic,n.args,n.bg,env_map=n.env)['render']
            identity=float((actual-baseline['rgb']).abs().max());maximum_identity=max(maximum_identity,identity);assert identity<=2e-6,identity
            if baseline_first is None:baseline_first=baseline['rgb'].clone();firstrow=row
            name=f'{row["raw"]:03d}'
            body=np.array(Image.open(n.source/'dynamic_masks/dynamic_mask_0'/(row['frame_id']+'.png')).resize((480,320),Image.Resampling.NEAREST))>0
            original_domain=binary_dilation(body,iterations=2)|(cpu(baseline['extra'])[0]>.05)
            for case in specs:
                out=baseline if case['family']=='no_edit' else n.render(case['frames'][i],case)
                is_camera=case['family'].startswith('camera')
                mask=np.zeros((320,480),dtype=bool) if is_camera or case['family']=='no_edit' else original_domain&~(cpu(out['extra'])[0]>.05)
                directory=a.output/case['id'];save(directory/'native_rgb'/(name+'.png'),out['rgb']);Image.fromarray(mask.astype('uint8')*255).save(directory/'mask'/(name+'.png'))
                np.savez_compressed(directory/'float'/(name+'.npz'),rgb=cpu(out['rgb']),alpha=cpu(out['alpha']),depth_sum=cpu(out['depth_sum']),dynamic_weight=cpu(out['extra']))
                records.append(dict(case=case['id'],raw=row['raw'],identity_max_difference=identity,requested_collision=bool(case['frames'][i]['request_box_collision_tracks']),stratum=case['request_stratum'],generation_pixels=int(mask.sum()),camera_generation_mask_status='pending_unknown_domain_native_only' if is_camera else 'old_support_disocclusion_new_object_protected',counterfactual_RGB_GT=False))
            print('MOTION_NATIVE',row['frame_id'],round(time.monotonic()-started,2),flush=True)
        continuous=None if a.single_frame or a.only_turn else continuous_revisions(n,specs,suite,a.output/'continuous')
        final=n.render(firstrow,base)['rgb'];roundtrip=float((final-baseline_first).abs().max());assert roundtrip==0 and n.unchanged()
    result=dict(status='native_single_request_suite_complete',context=suite['context'],run=str(a.run),iteration=a.iteration,frames=1 if a.single_frame else 31,single_request_variants=len(specs),single_request_families=sorted(set(x['family'] for x in specs)),rows=records,continuous_sequence_executed=False,adapter_actual_native_max_difference=maximum_identity,master_roundtrip_difference=roundtrip,master_files_and_tensor_versions_unchanged=True,seconds=time.monotonic()-started,cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved(),new_training_updates=0,backend_not_run_here=True,quality_accepted=False,human_verdict=None,limitations=['single original dynamic actor; reuse clone is not an independently reconstructed second vehicle','rigid Gaussian means/covariance edit, inherited world-space SH appearance, no learned asset improvement','annotated request collisions are stress strata, not an algorithm failure count','no counterfactual RGB GT or certified novel-view road completion','continuous multi-request revision suite still pending'])
    result['continuous_sequence_executed']=continuous is not None
    result['SH_orientation_convention']=a.sh_frame
    result['relighting_claim']=False
    result['continuous_revision_count']=continuous['revision_count'] if continuous else 0
    if continuous:result['limitations'][-1]='continuous deterministic native revisions executed; fixed video/backend comparison and learned memory benefit remain pending'
    (a.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rows'}))

if __name__=='__main__':
    from gpu_slot import exclusive
    with exclusive():main()
