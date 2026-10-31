"""R4固定表示共享绑定：真实V42 CUDA光栅，独立参考不进入此进程。"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k,'1')
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
from PIL import Image
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
RECON=HERE.parent/'sim_mechanism_20261030/reconstruction'
sys.path.insert(0,str(RECON))
from native_worker import SOURCE,assert_method,bind_scene_contract
from r4_core import canonical_hash,enumerate_routes,freeze_family

HELPER=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v41_20261023/engine/utils/v4_road_render.py')
OPERATIONS=('none','delete','shift','rotate')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    target=Path(path);temporary=target.with_suffix(target.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(target)


def file_hash(path):
    digest=sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(4*1024*1024),b''):digest.update(data)
    return digest.hexdigest()


def equal_state(left,right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left.cpu(),right.cpu())
    if isinstance(left,np.ndarray) or isinstance(right,np.ndarray):
        return isinstance(left,np.ndarray) and isinstance(right,np.ndarray) and np.array_equal(left,right)
    if isinstance(left,dict) and isinstance(right,dict):
        return set(left)==set(right) and all(equal_state(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)) and isinstance(right,(list,tuple)):
        return len(left)==len(right) and all(equal_state(a,b) for a,b in zip(left,right))
    return left==right


def save_rgb(path,array):
    data=np.round(np.clip(np.moveaxis(array,0,-1),0,1)*255).astype(np.uint8)
    Image.fromarray(data).save(path)


def left_quaternion(rotation,original):
    value=torch.tensor(np.roll(Rotation.from_matrix(rotation).as_quat(),1),dtype=original.dtype,device=original.device)
    aw,ax,ay,az=value.unbind(-1);bw,bx,by,bz=original.unbind(-1)
    result=torch.stack((aw*bw-ax*bx-ay*by-az*bz,aw*bx+ax*bw+ay*bz-az*by,
        aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw),-1)
    return result/result.norm(dim=-1,keepdim=True).clamp_min(1e-12)


def take_rows(values,indices):
    return {key:None if value is None else value[indices] for key,value in values.items()}


class NativeScene:
    def __init__(self,cpu_scene_dir,output,source=SOURCE):
        self.cpu=Path(cpu_scene_dir);self.output=Path(output);self.output.mkdir(parents=True)
        self.candidate=read(self.cpu/'candidate_manifest.json')
        self.family=freeze_family(self.candidate);self.routes=read(self.cpu/'shared_routes.json')
        if self.routes!=enumerate_routes(self.candidate):raise ValueError('冻结路由域不一致')
        self.allrebind_id='h%03d'%(len(self.routes)-1)
        self.tasks=read(self.cpu/'tasks.json');self.frozen=self.tasks[0]['frozen_scene_context']
        self.context=read(self.cpu/'camera_track_context.json')
        self.track={row['frame_id']:row for row in self.context['frames']}
        self.frames=list(self.track)
        self.anchors=[self.frames[i] for i in sorted({0,len(self.frames)//2,len(self.frames)-1})]
        self.scene_id=self.candidate['scene_id']
        contract=read(self.cpu.parent/'run_contract.json')
        self.event_path=Path(next(x['event'] for x in contract['checkpoint_inputs'] if x['id']==self.scene_id))
        self.event=read(self.event_path);self.config=self.event['config']
        if file_hash(self.event_path)!=self.frozen['event_sha256']:raise ValueError('event冻结身份改变')
        self.original_files={}
        for name,row in self.event['checkpoints'].items():
            path=Path(row['path']);stat=path.stat()
            if stat.st_size!=row['bytes'] or file_hash(path)!=self.frozen['component_sha256'][name]:
                raise ValueError('完整checkpoint bundle身份不一致:'+name)
            self.original_files[str(path)]=(stat.st_size,stat.st_mtime_ns)
        self.scene_root=Path(self.event['source_path'])
        for name,path,key in [('gauge',self.scene_root/'transform_matrix.txt','gauge_sha256'),
                ('preparation',self.scene_root/'preparation_manifest.json','preparation_sha256'),
                ('camera',Path(self.event['model_path'])/'cameras.json','camera_sha256')]:
            if file_hash(path)!=self.frozen[key]:raise ValueError(name+'冻结身份改变')
        self.transform=np.loadtxt(self.scene_root/'transform_matrix.txt')
        if not np.allclose(self.transform,np.asarray(self.context['world_to_model']),atol=1e-10):
            raise ValueError('CPU context和实际训练gauge不符')
        self.linear=self.transform[:3,:3];self.inverse_linear=np.linalg.inv(self.linear)
        self.gauge_scale=float(np.linalg.norm(self.linear[:,0]))
        assert_method(self.config)
        source=Path(source).resolve();sys.path.insert(0,str(source))
        from scene import Scene
        from scene.gaussian_model import GaussianModel,Dynamic_GaussianModel
        from scene.envlight import EnvLight
        from gaussian_renderer import prepare_rasterization,render
        if Path(sys.modules['scene'].__file__).resolve()!=source/'scene/__init__.py':raise ValueError('错误native source')
        self.prepare=prepare_rasterization;self.actual_render=render
        if not torch.cuda.is_available():raise RuntimeError('必须真实CUDA，不用toy renderer')
        torch.cuda.set_device(0);torch.set_num_threads(1)
        loader=self.output/'scene_loader';loader.mkdir()
        # 原Waymo loader会写transform_matrix；完全复制输入，禁止对原source做任何隐式写入。
        query_scene=loader/'query_scene';shutil.copytree(self.scene_root,query_scene,symlinks=False)
        if (query_scene/'transform_matrix.txt').is_symlink() or os.stat(query_scene/'transform_matrix.txt').st_ino==os.stat(self.scene_root/'transform_matrix.txt').st_ino:
            raise ValueError('私有gauge必须独立文件，不能hardlink/symlink')
        self.pipe=OmegaConf.create(dict(self.config,model_path=str(loader),source_path=str(query_scene),start_checkpoint=None))
        if self.pipe.compute_cov3D_python or self.pipe.convert_SHs_python or self.pipe.white_background:
            raise ValueError('只接受继承原生quaternion/SH/black-background配置')
        self.static=GaussianModel(self.pipe);self.dynamic=Dynamic_GaussianModel(self.pipe)
        init_s,init_d=self.static.create_from_pcd,self.dynamic.initialize
        self.static.create_from_pcd=lambda *_:None;self.dynamic.initialize=lambda *_:None
        proxy=type('R4BindingProxy',(),{'Scene':Scene})
        bind_scene_contract(proxy,dict(self.config,model_path=str(loader)))
        self.scene=proxy.Scene(self.pipe,self.static,self.dynamic,shuffle=False)
        self.static.create_from_pcd=init_s;self.dynamic.initialize=init_d
        self.environment=EnvLight(resolution=self.pipe.env_map_res).cuda()
        self.components=dict(static=self.static,dynamic=self.dynamic,environment=self.environment)
        for name,model in self.components.items():
            payload,step=torch.load(self.event['checkpoints'][name]['path'],map_location='cpu',mmap=True,weights_only=False)
            if step!=self.event['iteration']:raise ValueError('checkpoint step mismatch')
            payload=list(payload)
            if name=='environment':payload[0]=torch.nn.Parameter(payload[0].detach().cuda(),requires_grad=False)
            else:
                for i in range(1,11):payload[i]=payload[i].detach().cuda()
                model.time_duration=tuple(self.config['time_duration'])
            model.restore(tuple(payload),training_args=None)
            if getattr(model,'optimizer',None) is not None:raise ValueError('评价不创建optimizer')
            for value in vars(model).values():
                if torch.is_tensor(value):value.requires_grad_(False)
            del payload;gc.collect()
        self.continuation=torch.load(self.event['checkpoints']['training_state']['path'],map_location='cpu',weights_only=False)
        for name,attributes in self.continuation['model_runtime'].items():
            for key,value in attributes.items():setattr(self.components[name],key,value)
        self.scene.scale_index=self.continuation['scale_index']
        random.setstate(self.continuation['python_rng']);np.random.set_state(self.continuation['numpy_rng'])
        torch.set_rng_state(self.continuation['torch_rng']);torch.cuda.set_rng_state_all(self.continuation['cuda_rng'])
        spec=importlib.util.spec_from_file_location('r4_actual_native_raster',HELPER)
        self.helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.helpers)
        self.cameras={c.image_name:c for c in self.scene.getTrainCameras()}
        recorded={r['img_name']:r for r in read(Path(self.event['model_path'])/'cameras.json')}
        if set(self.cameras)!=set(recorded) or set(self.frames)!=set(recorded):raise ValueError('相机/时间总分母不符')
        emax,kmax=0.,0.;camera_recoveries=[]
        for name,camera in self.cameras.items():
            row=recorded[name];c2w=np.eye(4);c2w[:3,:3]=row['rotation'];c2w[:3,3]=row['position']
            raw_error=float(np.max(np.abs(camera.c2w.cpu().numpy()-c2w)))
            if raw_error>1e-5:
                # 历史hybrid训练锁定M0 gauge；重载器会从新相机轨迹重算归一化。
                # checkpoint保存的训练camera才是冻结状态，不可用新归一化相机查询旧模型。
                from utils.graphics_utils import getProjectionMatrixCenterShift
                world_to_camera=np.linalg.inv(c2w)
                camera.R=world_to_camera[:3,:3].T;camera.T=world_to_camera[:3,3]
                camera.trans=np.zeros(3);camera.scale=1.
                dx=row['width']/camera.image_width;dy=row['height']/camera.image_height
                camera.fx=row['fx']/dx;camera.fy=row['fy']/dy;camera.cx=row['cx']/dx;camera.cy=row['cy']/dy
                camera.world_view_transform=torch.tensor(world_to_camera,dtype=torch.float32,device='cuda').T.contiguous()
                camera.projection_matrix=getProjectionMatrixCenterShift(camera.znear,camera.zfar,camera.cx,camera.cy,
                    camera.fx,camera.fy,camera.image_width,camera.image_height).T.cuda()
                camera.full_proj_transform=(camera.world_view_transform.unsqueeze(0).bmm(camera.projection_matrix.unsqueeze(0))).squeeze(0)
                camera.camera_center=camera.world_view_transform.inverse()[3,:3]
                camera.c2w=camera.world_view_transform.T.inverse()
                camera_recoveries.append(dict(frame=name,loader_c2w_max_abs_before=raw_error,
                    restored_from='verified_frozen_training_cameras_json',source_files_changed=False))
            emax=max(emax,float(np.max(np.abs(camera.c2w.cpu().numpy()-c2w))))
            dx=row['width']/camera.image_width;dy=row['height']/camera.image_height
            kmax=max(kmax,abs(float(camera.fx)-row['fx']/dx),abs(float(camera.fy)-row['fy']/dy),
                abs(float(camera.cx)-row['cx']/dx),abs(float(camera.cy)-row['cy']/dy))
            if abs(float(camera.timestamp)-self.track[name]['model_time'])>1e-7:raise ValueError('合法track模型时间不符')
        if emax>1e-5 or kmax>1e-5:raise ValueError('native相机恢复失败')
        self.actor_source_check=self.check_actor_source()
        write(self.output/'native_restore_receipt.json',dict(status='RESTORED_ORIGINAL_NATIVE_BUNDLE',
            checkpoint_bundle_sha256=self.frozen['checkpoint_bundle_sha256'],components=list(self.components),
            model_runtime_restored=True,original_training_rng_restored=True,scene_scale_index=self.scene.scale_index,
            native_optimizer_created=False,training_updates=0,camera_count=len(self.cameras),
            extrinsic_max_abs=emax,intrinsic_max_abs=kmax,actor_source_check=self.actor_source_check,
            explicit_frozen_camera_recoveries=camera_recoveries,
            source_loader_isolation=dict(private_scene=str(query_scene),all_input_files_independently_copied=True,
                transform_is_independent_copy=True,original_source_never_passed_to_mutating_loader=True,
                private_loader_transform_sha256=file_hash(query_scene/'transform_matrix.txt'),
                active_checkpoint_gauge_sha256=self.frozen['gauge_sha256']),
            environment_restored=True,source=str(source),native_python=sys.executable,
            original_training_optimizer_state='retained_in_verified_frozen_checkpoint_not_instantiated_for_inference'))
        self.base_background=torch.zeros(3,device='cuda')
        self.versions_before=self.versions()
        self.groups={g['group_id']:g for g in self.candidate['groups']}
        self.group_indices={gid:torch.tensor(g['original_static_rows'],dtype=torch.long,device='cuda') for gid,g in self.groups.items()}
        self.source_versions=dict(context_sha256=file_hash(self.cpu/'camera_track_context.json'),
            candidate_sha256=file_hash(self.cpu/'candidate_manifest.json'),route_manifest_sha256=file_hash(self.cpu/'shared_routes.json'))

    def check_actor_source(self):
        spec=self.config['followup_scene_spec'];paths=spec['actor_ply_paths']
        if len(paths)!=1:raise ValueError('多actor native动态行映射未知，不得假定整个dynamic=actor0')
        record=dict(native_dynamic_actor_source_count=1,target_id=self.candidate['target_id'],actor_ply_paths=paths,
            role='inherited_single_actor_branch_binding_not_per_Gaussian_owner_GT')
        path=self.scene_root/'dynamic_multi_actor_preparation.json'
        if path.is_file():
            data=read(path)
            if len(data['actors'])!=1 or data['actors'][0]['actor_index']!=0:raise ValueError('真实源不是单actor0')
            actual_track=data['actors'][0]['raw_track_id']
            legal_manifest=read(self.context['role']['original_input_contract'])
            review_path=self.context['role']['source_identity_review'];review=read(review_path)
            if actual_track!=legal_manifest['track'] or actual_track!=review['track']:
                raise ValueError('native dynamic源track与CPU合法目标或原始关联review不一致')
            record.update(source_preparation=str(path),raw_track_id=actual_track,
                cpu_legal_target_track=legal_manifest['track'],original_review_track=review['track'],
                exact_target_track_assertion_passed=True)
        return record

    def versions(self):
        return {(name,key):(v._version,tuple(v.shape)) for name,model in self.components.items()
            for key,v in vars(model).items() if torch.is_tensor(v)}

    def unchanged(self):
        after={name:(Path(name).stat().st_size,Path(name).stat().st_mtime_ns) for name in self.original_files}
        if after!=self.original_files or self.versions()!=self.versions_before:raise ValueError('原模型参数或checkpoint被修改')
        return True

    def frame_rows(self,frame):
        camera=self.cameras[frame]
        static=self.prepare(camera,self.static,self.pipe);dynamic=self.prepare(camera,self.dynamic,self.pipe,dynamic=True)
        static['means2D']=torch.zeros_like(static['means3D']);dynamic['means2D']=torch.zeros_like(dynamic['means3D'])
        if len(static['means3D'])!=self.candidate['static_row_count']:raise ValueError('native static original row namespace不一致')
        env=self.environment(camera.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
        return camera,static,dynamic,env

    def update_depth(self,rows,camera):
        result=dict(rows)
        depth=torch.cat((rows['means3D'],torch.ones_like(rows['means3D'][:,:1])),1)@camera.world_view_transform[:,2:3]
        result['features']=torch.cat((depth,torch.ones_like(depth)),1)
        return result

    def apply_shared_route(self,route,frame,static,dynamic):
        """不接受用户q：同一绑定首先生成当前帧真实场景，包括none。"""
        selected=[a['group_id'] for a in route['assignments'] if a['action']=='rebind']
        n=len(static['means3D']);mask=torch.zeros(n,dtype=torch.bool,device='cuda')
        if not selected:return static,dynamic,dict(original_static_rows=n,kept_static_rows=n,rebound_static_rows=0,
            native_dynamic_prepared_rows=len(dynamic['means3D']),preedit_total_rows=n+len(dynamic['means3D']),
            unique_original_static_partition=True,rows_added=0,rows_duplicated=0)
        current=self.track[frame];center=np.asarray(current['actor_center_world']);rotation=np.asarray(current['actor_rotation_world'])
        bound=[]
        for gid in selected:
            group=self.groups[gid];indices=self.group_indices[gid];mask[indices]=True
            rows=take_rows(static,indices)
            anchor_center=np.asarray(group['anchor_actor_center_world'])
            anchor_rotation=np.asarray(group['anchor_actor_rotation_world'])
            relative=self.linear@(rotation@anchor_rotation.T)@self.inverse_linear
            old=torch.tensor(self.linear@anchor_center+self.transform[:3,3],dtype=torch.float32,device='cuda')
            new=torch.tensor(self.linear@center+self.transform[:3,3],dtype=torch.float32,device='cuda')
            matrix=torch.tensor(relative,dtype=torch.float32,device='cuda')
            # 原生SHM残差/生命周期/SH参数保持；只复合合法actor的anchor→当前刚性绑定。
            rows['means3D']=(rows['means3D']-old)@matrix.T+new
            rows['rotations']=left_quaternion(relative,rows['rotations'])
            rows=self.update_depth(rows,self.cameras[frame]);bound.append(rows)
        kept=take_rows(static,~mask);moving=dynamic
        for rows in bound:moving=self.helpers.merge_rows(rows,moving)
        rebound=int(mask.sum())
        expected=sum(len(self.groups[g]['original_static_rows']) for g in selected)
        if expected!=rebound or len(kept['means3D'])+len(moving['means3D'])!=n+len(dynamic['means3D']):raise ValueError('原行守恒失败')
        return kept,moving,dict(original_static_rows=n,kept_static_rows=n-rebound,rebound_static_rows=rebound,
            native_dynamic_prepared_rows=len(dynamic['means3D']),preedit_total_rows=n+len(dynamic['means3D']),
            unique_original_static_partition=True,rows_added=0,rows_duplicated=0)

    def execute_query(self,frame,static,moving,edit):
        if edit not in OPERATIONS:raise ValueError('非法编辑')
        if edit=='delete':return static
        moving=dict(moving)
        if edit=='shift':
            delta=torch.tensor(self.linear@np.array([0.,1.5,0.]),dtype=torch.float32,device='cuda')
            moving['means3D']=moving['means3D']+delta
        elif edit=='rotate':
            relative=self.linear@Rotation.from_euler('z',30,degrees=True).as_matrix()@self.inverse_linear
            center_world=np.asarray(self.track[frame]['actor_center_world'])
            center=torch.tensor(self.linear@center_world+self.transform[:3,3],dtype=torch.float32,device='cuda')
            matrix=torch.tensor(relative,dtype=torch.float32,device='cuda')
            moving['means3D']=(moving['means3D']-center)@matrix.T+center
            moving['rotations']=left_quaternion(relative,moving['rotations'])
        if edit!='none':moving=self.update_depth(moving,self.cameras[frame])
        return self.helpers.merge_rows(static,moving)

    def raster(self,frame,rows,env):
        package=self.helpers.raster(self.cameras[frame],self.pipe,self.static.active_sh_degree,rows,self.base_background)
        color,alpha=package['rgb'],package['alpha'];fill=(1-alpha)*env
        values=dict(C_G=color,alpha=alpha,E=env,fill=fill,final=color+fill,white=color+(1-alpha),depth_sum_model=package['depth_sum'])
        result={k:v.detach().cpu().numpy().astype(np.float32) for k,v in values.items()}
        if any(not np.isfinite(v).all() for v in result.values()):raise FloatingPointError('native FP32非有限')
        return result

    def native_baseline(self,frame,static,dynamic,env):
        arrays=self.raster(frame,self.helpers.merge_rows(static,dynamic),env)
        actual=self.actual_render(self.cameras[frame],self.static,self.dynamic,self.pipe,self.base_background,env_map=self.environment)['render']
        error=float(np.max(np.abs(actual.detach().cpu().numpy()-arrays['final'])))
        if error>2e-6:raise RuntimeError('原native none identity失败:'+str(error))
        return arrays,error

    def save_state(self,path,route_id,smoke_manifest):
        state=dict(schema='r4_complete_inference_state_v1',cpu_scene_dir=str(self.cpu),scene_id=self.scene_id,
            checkpoint_event=str(self.event_path),frozen_scene_context=self.frozen,source_versions=self.source_versions,
            routes=self.routes,candidate_hash=self.family['candidate_hash'],route_id=route_id,
            native_bundle_refs=self.event['checkpoints'],model_runtime=self.continuation['model_runtime'],
            native_training_state_ref=self.event['checkpoints']['training_state']['path'],
            native_training_state=self.continuation,scene_scale_index=self.scene.scale_index,
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            smoke_manifest=str(smoke_manifest),training_updates=0,
            state_storage='complete inference state plus verified immutable original tensor/optimizer bundle references; no model copy or update')
        torch.save(state,path)
        reload=torch.load(path,map_location='cpu',weights_only=False)
        if reload['frozen_scene_context']!=self.frozen or reload['routes']!=self.routes:raise RuntimeError('保存state重读失败')
        return dict(path=str(path),bytes=Path(path).stat().st_size,sha256=file_hash(path),serialization_roundtrip_passed=True,
            parameter_storage='verified_original_frozen_bundle_refs',contains_all_native_training_state_metadata=True)

    def restore_state(self,path):
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved['schema']!='r4_complete_inference_state_v1' or saved['frozen_scene_context']!=self.frozen:
            raise ValueError('完整恢复state指向不同frozen scene')
        if saved['routes']!=self.routes or saved['source_versions']!=self.source_versions:raise ValueError('route/camera-track冻结输入改变')
        if not equal_state(saved['model_runtime'],self.continuation['model_runtime']) or saved['scene_scale_index']!=self.scene.scale_index:
            raise ValueError('motion/environment/scale runtime恢复不匹配')
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        return saved


def save_arrays(folder,frame,arrays):
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/(frame+'.npz');np.savez_compressed(path,**arrays)
    for name in ('final','white'):save_rgb(folder/(frame+'_'+name+'.png'),arrays[name])
    return str(path)


def render_rows(native,output,frames,route_ids,compare_native=True):
    result=[];baseline=[];parity=0.;start=time.monotonic()
    pool=ThreadPoolExecutor(max_workers=2);jobs=[]
    with torch.no_grad():
        for frame in frames:
            camera,static,dynamic,env=native.frame_rows(frame)
            original,error=native.native_baseline(frame,static,dynamic,env) if compare_native else (None,0.)
            if compare_native:
                path=save_arrays(output/'native_original_none',frame,original)
                baseline.append(dict(frame=frame,camera_id='image_0/'+frame,fp32=path,native_none_identity_max_abs=error))
                parity=max(parity,error)
            for action_id in route_ids:
                index=int(action_id[1:]);route=native.routes[index]
                static_bound,moving,conservation=native.apply_shared_route(route,frame,static,dynamic)
                route_seal=canonical_hash(route)
                for edit in OPERATIONS:
                    before=time.monotonic()
                    rows=native.execute_query(frame,static_bound,moving,edit)
                    arrays=native.raster(frame,rows,env)
                    if edit=='none' and action_id=='h000' and original is not None:
                        difference=max(float(np.max(np.abs(arrays[k]-original[k]))) for k in arrays)
                        parity=max(parity,difference)
                        if difference>2e-6:raise RuntimeError('allkeep共享none与原native不一致')
                    if canonical_hash(route)!=route_seal:raise ValueError('query更改共享route')
                    folder=output/'routes'/action_id/edit;fp32=str(folder/(frame+'.npz'))
                    jobs.append(pool.submit(save_arrays,folder,frame,arrays))
                    if len(jobs)>4:jobs.pop(0).result()
                    result.append(dict(scene_id=native.scene_id,action_id=action_id,
                        route_assignment_hash=route['route_assignment_hash'],frame=frame,edit=edit,
                        camera_id='image_0/'+frame,model_time=float(camera.timestamp),fp32=fp32,
                        actual_shared_route_applied=True,none_is_checkpoint_bypass=False,
                        row_conservation=dict(conservation,query_rows=len(rows['means3D']),
                            query_omitted_rows=conservation['preedit_total_rows']-len(rows['means3D'])),
                        compositor_checks=dict(final_vs_C_plus_fill_max_abs=float(np.max(np.abs(arrays['final']-(arrays['C_G']+arrays['fill'])))),
                            fill_vs_one_minus_alpha_E_max_abs=float(np.max(np.abs(arrays['fill']-(1-arrays['alpha'])*arrays['E'])))),
                        gpu_render_and_readback_wall_seconds=time.monotonic()-before))
                del static_bound,moving
                print(json.dumps(dict(scene=native.scene_id,frame=frame,action=action_id,completed_rows=len(result))),flush=True)
    for job in jobs:job.result()
    pool.shutdown()
    return result,baseline,parity,time.monotonic()-start


def manifest(native,rows,baseline,parity,wall,phase):
    return dict(schema='r4_native_shared_route_render_v1',status='NATIVE_SHARED_ROUTE_RENDERED',phase=phase,
        scene_id=native.scene_id,cpu_frozen_scene_dir=str(native.cpu),frozen_scene_context=native.frozen,
        candidate_hash=native.family['candidate_hash'],shared_route_context_hash=native.tasks[0]['shared_route_context_hash'],
        checkpoint_bundle_sha256=native.frozen['checkpoint_bundle_sha256'],source_checkpoint_event=str(native.event_path),
        scene_ancestor=native.candidate['scene_ancestor'],frame_total_denominator=len(native.frames),
        registered_frame_ids=native.frames,evaluated_anchor_frames=native.anchors,remaining_frames_status='unmeasured_null',
        edits=list(OPERATIONS),action_count=len(native.routes),allkeep_action_id='h000',allrebind_action_id=native.allrebind_id,
        none_uses_actual_route=True,allkeep_native_none_parity_max_abs=parity,native_original_none=baseline,rows=rows,
        route_edit_render_calls=len(rows),baseline_helper_control_render_calls=len(baseline),actual_native_control_render_calls=len(baseline),
        route_semantics='frozen per-group anchor-to-current legal rigid actor transform composed with original native SHM row state; query is applied only after shared binding',
        inherited_SH_frame='world-space native SH, no new relighting',geometry_color_opacity_capacity_environment_parameters_frozen=True,
        target_selection='current CPU registered actor0, not a global ranking or all actors in scene',
        original_checkpoint_unchanged=native.unchanged(),training_updates=0,
        independent_reference_files_read_by_renderer=False,human_verdict=None,
        wall_seconds=wall,cuda_max_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
        cuda_device_name=torch.cuda.get_device_name(),eligibility_status='CPU_legal_preedit_uncalibrated_unknown',
        decision_gap=None,capability_residual=None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-scene-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=['smoke','enumerate'],required=True)
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,args.output/'native_executable_snapshot.py')
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    native=NativeScene(args.cpu_scene_dir,args.output/'restore')
    write(args.output/'render_contract.json',dict(cpu_scene_dir=str(args.cpu_scene_dir),phase=args.phase,
        candidates_and_routes='exact CPU frozen manifests',frame_total_denominator=len(native.frames),
        evaluated_anchor_frames=native.anchors,operations=list(OPERATIONS),possible_routes=len(native.routes),
        representation_parameters='fixed',reference_reading='forbidden in this executable',
        original_other_frames='registered but unmeasured; preserve null',target='CPU registered actor0 only'))
    if args.phase=='smoke':
        frames=[native.anchors[1]]
        rows,baseline,parity,wall=render_rows(native,args.output,frames,list(dict.fromkeys(['h000',native.allrebind_id])))
        result=manifest(native,rows,baseline,parity,wall,'smoke')
        write(args.output/'gpu_manifest.json',result)
        checkpoint=native.save_state(args.output/'inference_state.pt',native.allrebind_id,args.output/'gpu_manifest.json')
        write(args.output/'saved_state_receipt.json',checkpoint)
        write(args.output/'next_stage_entry.json',dict(schema='r4_next_stage_entry_v1',phase='enumerate',
            checkpoint=checkpoint,cpu_scene_dir=str(args.cpu_scene_dir),resume_required=True,
            next_stage='fresh process original complete bundle restore plus saved route/runtime/RNG and smoke replay before enumeration'))
    else:
        if args.resume is None:raise ValueError('必须经过真实smoke保存/恢复入口')
        state=native.restore_state(args.resume)
        prior=read(state['smoke_manifest']);frame=native.anchors[1]
        replay,_,_,_=render_rows(native,args.output/'restore_replay',[frame],list(dict.fromkeys(['h000',native.allrebind_id])))
        errors=[]
        for row in replay:
            old=next(x for x in prior['rows'] if all(x[k]==row[k] for k in ('action_id','frame','edit')))
            with np.load(old['fp32'],allow_pickle=False) as a,np.load(row['fp32'],allow_pickle=False) as b:
                error=max(float(np.max(np.abs(a[k]-b[k]))) for k in a.files)
            errors.append(dict(action_id=row['action_id'],frame=row['frame'],edit=row['edit'],fp32_max_abs=error))
        passed=all(x['fp32_max_abs']<=2e-6 for x in errors)
        write(args.output/'complete_restore_replay_receipt.json',dict(all_passed=passed,rows=errors,
            native_bundle_all_components_restored=True,camera_motion_environment_runtime_restored=True,
            route_and_RNG_restored=True,saved_state=str(args.resume),next_stage_entry_exercised=True,training_updates=0,
            route_edit_render_calls=len(replay),baseline_helper_control_render_calls=1,actual_native_control_render_calls=1))
        if not passed:raise RuntimeError('完整保存/新进程恢复的FP32重放不符')
        rows,baseline,parity,wall=render_rows(native,args.output,native.anchors,['h%03d'%i for i in range(len(native.routes))])
        result=manifest(native,rows,baseline,parity,wall,'complete_8_route_enumeration')
        write(args.output/'gpu_manifest.json',result)
    write(args.output/'execution.json',dict(status='COMPLETE',phase=args.phase,wall_seconds=time.monotonic()-start,
        gpu_rendered_rows=len(result['rows']),training_updates=0,original_checkpoint_unchanged=native.unchanged(),
        max_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),max_cuda_reserved_bytes=torch.cuda.max_memory_reserved()))
    print(json.dumps(dict(status='COMPLETE',scene_id=native.scene_id,phase=args.phase,output=str(args.output),
        rows=len(result['rows']),wall_seconds=time.monotonic()-start)),flush=True)


if __name__=='__main__':main()
