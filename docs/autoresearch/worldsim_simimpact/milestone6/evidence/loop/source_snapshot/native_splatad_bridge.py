"""官方 SplatAD 传感器桥：保存标定，按实际 ego 位姿重新渲染。"""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation, Slerp
from splatad_compat import apply
apply()
from nerfstudio.utils.eval_utils import eval_setup

CAMERAS=('CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT')
CV_TO_NS=np.diag([1.,-1.,-1.,1.])

class SceneMetadata:
    def __init__(self,path):self.tables=json.loads(path.read_text())
    def get(self,table,token):return self.tables[table][token]

def matrix(rec):
    t=np.eye(4);t[:3,:3]=Quaternion(rec['rotation']).rotation_matrix
    t[:3,3]=rec['translation'];return t

def homogeneous(t):
    r=np.eye(4);r[:3,:]=np.asarray(t);return r

def to_numpy(t):
    return t.detach().cpu().numpy()

class NativeSplatADBridge:
    def __init__(self,fit:Path,checkpoint:Path,step:int,initial_sample:str):
        def update(c):
            c.load_dir=checkpoint;c.load_step=step
            c.pipeline.datamanager.cache_images='cpu';c.pipeline.datamanager.cache_lidars='cpu'
            c.pipeline.datamanager.max_thread_workers=4;c.pipeline.calc_fid_steps=()
            return c
        self.config,self.pipeline,_,self.step=eval_setup(fit/'config.yml',test_mode='val',update_config_callback=update)
        self.dm=self.pipeline.datamanager;self.model=self.pipeline.model
        self.model.eval();self.device=self.model.device
        self.parser=self.dm.dataparser
        self.data_root=self.config.pipeline.datamanager.dataparser.data
        self.nusc=SceneMetadata(self.data_root.parent/'metadata'/f'{self.parser.config.sequence}.json')
        outputs=self.dm.train_dataparser_outputs
        self.A=homogeneous(to_numpy(outputs.dataparser_transform))
        self.time_offset=outputs.time_offset
        self.sample=self.nusc.get('sample',initial_sample)
        self.extrinsics={};self.templates={};self.camera_checks={}
        self.train_images={str(p) for p in self.dm.train_dataset.image_filenames}
        self.eval_images={str(p) for p in self.dm.eval_dataset.image_filenames}
        for name in CAMERAS:
            sd=self.nusc.get('sample_data',self.sample['data'][name])
            self.extrinsics[name]=matrix(self.nusc.get('calibrated_sensor',sd['calibrated_sensor_token']))@CV_TO_NS
            filename=str(self.data_root/sd['filename'])
            dataset=self.dm.train_dataset if filename in self.train_images else self.dm.eval_dataset
            i=list(map(str,dataset.image_filenames)).index(filename)
            template=copy.deepcopy(dataset.cameras[i:i+1]).to(self.device)
            template.metadata['cam_idx']=i
            self.templates[name]=template
            world_pose=self.ego_pose(sd)@self.extrinsics[name]
            err=np.max(abs(self.A@world_pose-homogeneous(to_numpy(template.camera_to_worlds)[0])))
            self.camera_checks[name]={'max_calibration_matrix_difference':float(err),
                'split':'train' if filename in self.train_images else 'heldout'}
            assert err<.001, (name,err)

        sd=self.nusc.get('sample_data',self.sample['data']['LIDAR_TOP'])
        self.extrinsics['LIDAR_TOP']=matrix(self.nusc.get('calibrated_sensor',sd['calibrated_sensor_token']))
        lidar_time=sd['timestamp']/1e6-self.time_offset
        options=[]
        for split,dataset in [('train',self.dm.train_lidar_dataset),('eval',self.dm.eval_lidar_dataset)]:
            times=to_numpy(dataset.lidars.times).reshape(-1);idx=int(abs(times-lidar_time).argmin())
            options.append((abs(times[idx]-lidar_time),split,dataset,idx))
        diff,split,dataset,idx=min(options,key=lambda q:q[0]);assert diff<1e-4
        self.lidar=copy.deepcopy(dataset.lidars[idx:idx+1]).to(self.device)
        data=copy.deepcopy((self.dm.cached_lidar_train if split=='train' else self.dm.cached_lidar_eval)[idx])
        data['is_eval']=True
        self.lidar.metadata['lidar_idx']=idx
        self.dm._add_metadata(self.lidar,data,len(self.dm.train_dataset if split=='train' else self.dm.eval_dataset))
        self.raster=data['raster_pts'].clone()
        self.pattern_valid=self.raster[...,2]>0
        self.raster[...,2]=self.pattern_valid.float()
        self.raster[...,4]=0
        self.lidar.metadata['raster_pts']=self.raster
        self.lidar_check={'initial_log_split':split,'valid_firing_directions':int(self.pattern_valid.sum()),
            'pattern_shape':list(self.raster.shape),'time_min_s':float(self.raster[...,3][self.pattern_valid].min()),
            'time_max_s':float(self.raster[...,3][self.pattern_valid].max()),
            'scope':'Fixed firing directions/times calibrated from the initial real scan, including official imputed directions and ego exclusion. No future GT ranges or validity masks; not an ideal uniform manufacturer scan.'}
        angles=torch.deg2rad(self.raster[...,:2])
        self.directions=torch.stack([angles[...,1].cos()*angles[...,0].cos(),
            angles[...,1].cos()*angles[...,0].sin(),angles[...,1].sin()],-1)
        # 保留完整日志姿态用于 teacher-forced 参考；闭环传感器不读取未来姿态。
        self.log_samples=[];s=self.nusc.get('scene',self.sample['scene_token'])['first_sample_token']
        while s:
            sample=self.nusc.get('sample',s);self.log_samples.append(sample);s=sample['next']
        self.front_records=[self.nusc.get('sample_data',s['data']['CAM_FRONT']) for s in self.log_samples]
        self.front_times=np.array([s['timestamp']/1e6 for s in self.front_records])
        self.front_poses=np.array([self.ego_pose(s) for s in self.front_records])
        self.rotation_interp=Slerp(self.front_times,Rotation.from_matrix(self.front_poses[:,:3,:3]))

    def ego_pose(self,sd):
        return matrix(self.nusc.get('ego_pose',sd['ego_pose_token']))

    def pose_at(self,t):
        assert self.front_times[0]<=t<=self.front_times[-1]
        pose=np.eye(4);pose[:3,:3]=self.rotation_interp([t]).as_matrix()[0]
        pose[:3,3]=[np.interp(t,self.front_times,self.front_poses[:,i,3]) for i in range(3)]
        return pose

    def status_at(self,sd):
        seq=[sd]
        for _ in range(2):
            if not seq[-1]['prev']:break
            seq.append(self.nusc.get('sample_data',seq[-1]['prev']))
        if len(seq)<3:raise RuntimeError('Causal state needs two earlier sensor records')
        poses=[self.ego_pose(s) for s in seq]
        dt=[(seq[i]['timestamp']-seq[i+1]['timestamp'])/1e6 for i in range(2)]
        vg=[(poses[i][:3,3]-poses[i+1][:3,3])/dt[i] for i in range(2)]
        v=poses[0][:3,:3].T@vg[0]
        a=poses[0][:3,:3].T@(2*(vg[0]-vg[1])/(dt[0]+dt[1]))
        omega=Rotation.from_matrix(poses[1][:3,:3].T@poses[0][:3,:3]).as_rotvec()/dt[0]
        return v,a,omega

    def _sensor(self,name,ego_pose,t,v,omega):
        template=self.lidar if name=='LIDAR_TOP' else self.templates[name]
        sensor=copy.deepcopy(template)
        ext=self.extrinsics[name]
        transform=torch.tensor((self.A@ego_pose@ext)[:3],dtype=torch.float32,device=self.device)[None]
        if name=='LIDAR_TOP':sensor.lidar_to_worlds=transform
        else:sensor.camera_to_worlds=transform
        sensor.times=torch.tensor([[t-self.time_offset]],dtype=sensor.times.dtype,device=self.device)
        # 含传感器杆臂转动速度；本地速度用于官方 rolling shutter。
        local_v=ext[:3,:3].T@(np.asarray(v)+np.cross(omega,ext[:3,3]))
        local_w=ext[:3,:3].T@np.asarray(omega)
        sensor.metadata['linear_velocities_local']=torch.tensor(local_v,dtype=torch.float32,device=self.device)[None]
        sensor.metadata['angular_velocities_local']=torch.tensor(local_w,dtype=torch.float32,device=self.device)[None]
        return sensor

    @torch.inference_mode()
    def camera(self,name,ego_pose,t,v,omega):
        sensor=self._sensor(name,ego_pose,t,v,omega)
        output=self.model.get_camera_outputs(sensor)
        rgb=np.clip(to_numpy(output['rgb'])*255,0,255).astype(np.uint8)
        return rgb,to_numpy(sensor.camera_to_worlds)[0]

    @torch.inference_mode()
    def lidar_scan(self,ego_pose,t,v,omega,placeholder=1.):
        sensor=self._sensor('LIDAR_TOP',ego_pose,t,v,omega)
        sensor.metadata['raster_pts']=self.raster.clone()
        sensor.metadata['raster_pts'][...,2]=self.pattern_valid.float()*placeholder
        output=self.model.get_lidar_outputs(sensor)
        keep=self.pattern_valid&(output['ray_drop_prob'][...,0]<=.5)
        # 保持官方两条点云路径；使用官方渲染后居中的时间和相同预测 ray-drop。
        offsets=sensor.metadata['raster_pts'][...,3]
        motion=sensor.metadata['linear_velocities_local']*offsets[...,None]
        result={}
        ext=self.extrinsics['LIDAR_TOP']
        for name,key in [('raw','depth'),('median','median_depth')]:
            xyz=(output[key]*self.directions+motion)[keep]
            xyz=to_numpy(xyz)
            result[name]=(xyz@ext[:3,:3].T+ext[:3,3]).astype(np.float32)
        result.update(raw_range=to_numpy(output['depth']),median_range=to_numpy(output['median_depth']),
            keep=to_numpy(keep),ray_drop_prob=to_numpy(output['ray_drop_prob']),
            intensity=to_numpy(output['intensity'][...,0][keep]),
            lidar_to_world=to_numpy(sensor.lidar_to_worlds)[0],policy_lidar_origin_ego=ext[:3,3].copy())
        return result

    def render(self,ego_pose,t,v,omega):
        rgb={};poses={}
        for name in CAMERAS:rgb[name],poses[name]=self.camera(name,ego_pose,t,v,omega)
        scan=self.lidar_scan(ego_pose,t,v,omega)
        return {'rgb':rgb,'scan':scan,'camera_poses':poses,
            'ego_pose_world':np.asarray(ego_pose),'time_absolute_s':t,
            'sensor_times_absolute_s':{name:t for name in (*CAMERAS,'LIDAR_TOP')}}

    def render_log_sample(self,sample):
        """同位姿比较匹配各真实传感器采样时刻；仅用于日志评价分支。"""
        front=self.nusc.get('sample_data',sample['data']['CAM_FRONT'])
        query_ego=self.ego_pose(front);rgb={};poses={};times={}
        for name in CAMERAS:
            sd=self.nusc.get('sample_data',sample['data'][name]);ego=self.ego_pose(sd)
            v,_,w=self.status_at(sd);t=sd['timestamp']/1e6
            rgb[name],poses[name]=self.camera(name,ego,t,v,w);times[name]=t
        sd=self.nusc.get('sample_data',sample['data']['LIDAR_TOP'])
        ego=self.ego_pose(sd);v,_,w=self.status_at(sd);t=sd['timestamp']/1e6
        scan=self.lidar_scan(ego,t,v,w);times['LIDAR_TOP']=t
        relative=np.linalg.inv(query_ego)@ego
        for readout in ['raw','median']:
            scan[readout]=(scan[readout]@relative[:3,:3].T+relative[:3,3]).astype(np.float32)
        scan['policy_lidar_origin_ego']=relative[:3,:3]@scan['policy_lidar_origin_ego']+relative[:3,3]
        return {'rgb':rgb,'scan':scan,'camera_poses':poses,'ego_pose_world':query_ego,
            'time_absolute_s':front['timestamp']/1e6,'sensor_times_absolute_s':times}

    def real_observation(self,sample):
        from PIL import Image
        front=self.nusc.get('sample_data',sample['data']['CAM_FRONT'])
        ego=self.ego_pose(front);inv=np.linalg.inv(ego)
        rgb={name:np.array(Image.open(self.data_root/self.nusc.get('sample_data',sample['data'][name])['filename']).convert('RGB')) for name in CAMERAS}
        sd=self.nusc.get('sample_data',sample['data']['LIDAR_TOP'])
        transform=inv@self.ego_pose(sd)@self.extrinsics['LIDAR_TOP']
        points=np.fromfile(self.data_root/sd['filename'],np.float32).reshape(-1,5)[:,:3]
        points=(points@transform[:3,:3].T+transform[:3,3]).astype(np.float32)
        # 与既有真实扫描控制相同，保留 1–80 m；策略输入另使用相同范围规则。
        distances=np.linalg.norm(points-transform[:3,3],axis=1)
        points=points[np.isfinite(points).all(1)&(distances>1)&(distances<80)]
        return rgb,points,ego,front
