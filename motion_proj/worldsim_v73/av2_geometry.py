"""AV2 metric observations with nanosecond poses and per-return sensor origins.

Format sources: argoverse/av2-api structures/sweep.py and utils/io.py.
Sweeps are already ego-motion compensated into ego at the reference timestamp;
the city endpoint therefore uses that reference pose exactly once. Ray origins
and Actor canonical coordinates use each return's actual emission time.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp


RING_CAMERAS=('ring_front_center','ring_front_left','ring_front_right',
              'ring_side_left','ring_side_right','ring_rear_left','ring_rear_right')
RIGID_VEHICLES={'REGULAR_VEHICLE','LARGE_VEHICLE','BUS','BOX_TRUCK','TRUCK',
                'TRUCK_CAB','VEHICULAR_TRAILER','SCHOOL_BUS'}


def transforms(frame):
    matrices=np.broadcast_to(np.eye(4),(len(frame),4,4)).copy()
    matrices[:,:3,:3]=Rotation.from_quat(frame[['qx','qy','qz','qw']].to_numpy(float)).as_matrix()
    matrices[:,:3,3]=frame[['tx_m','ty_m','tz_m']].to_numpy(float)
    return matrices


class TimedPoses:
    """Interpolate known poses; validity is separate from clipped numeric values."""
    def __init__(self,timestamps_ns,matrices):
        self.times,indices=np.unique(np.asarray(timestamps_ns,dtype=np.int64),return_index=True)
        self.matrices=np.asarray(matrices)[indices]
        self.origin_ns=int(self.times[0])
        # Subtract int64 timestamps before float conversion to retain precision.
        self.seconds=(self.times-self.origin_ns).astype(np.float64)/1e9
        self.rotations=Slerp(self.seconds,Rotation.from_matrix(self.matrices[:,:3,:3])) if len(self.times)>1 else None

    def at(self,timestamps_ns):
        query=np.asarray(timestamps_ns,dtype=np.int64)
        flat=query.reshape(-1)
        valid=(flat>=self.times[0])&(flat<=self.times[-1])
        seconds=(np.clip(flat,self.times[0],self.times[-1])-self.origin_ns).astype(np.float64)/1e9
        result=np.broadcast_to(np.eye(4),(len(flat),4,4)).copy()
        result[:,:3,:3]=self.rotations(seconds).as_matrix() if self.rotations else self.matrices[0,:3,:3]
        for axis in range(3): result[:,axis,3]=np.interp(seconds,self.seconds,self.matrices[:,axis,3])
        return result.reshape(query.shape+(4,4)),valid.reshape(query.shape)


class AV2MetricLog:
    def __init__(self,log_dir):
        self.root=Path(log_dir)
        poses=pd.read_feather(self.root/'city_SE3_egovehicle.feather')
        self.ego=TimedPoses(poses.timestamp_ns.to_numpy(np.int64),transforms(poses))
        calibrated=pd.read_feather(self.root/'calibration/egovehicle_SE3_sensor.feather')
        self.ego_from_sensor=dict(zip(calibrated.sensor_name,transforms(calibrated)))
        self.intrinsics=pd.read_feather(self.root/'calibration/intrinsics.feather').set_index('sensor_name')
        self.tracks=None

    def load_tracks(self):
        # num_interior_pts is not used for selection or model input.
        annotations=pd.read_feather(self.root/'annotations.feather')
        tracks={}
        for owner,frame in annotations.groupby('track_uuid',sort=True):
            frame=frame.sort_values('timestamp_ns')
            times=frame.timestamp_ns.to_numpy(np.int64)
            ego,valid=self.ego.at(times)
            if not np.any(valid): continue
            city_from_actor=ego[valid]@transforms(frame.loc[valid])
            tracks[str(owner)]={'category':str(frame.category.iloc[0]),
                'poses':TimedPoses(times[valid],city_from_actor),
                'size_times_ns':times[valid],
                'sizes_lwh_m':frame.loc[valid,['length_m','width_m','height_m']].to_numpy(float)}
        self.tracks=tracks
        return tracks

    def sweep(self,filename,lower_ids_sensor='up_lidar'):
        """All observed returns, no density cap or synthetic no-return beams.

        The laser group convention is explicit; its evidence is recorded by the
        old-development physical calibration diagnostic, not inferred per test log.
        """
        path=Path(filename)
        if not path.is_absolute(): path=self.root/path
        frame=pd.read_feather(path)
        stamp=int(path.stem)
        point_times=stamp+frame.offset_ns.to_numpy(np.int64)
        reference,reference_valid=self.ego.at(stamp)
        ego,pose_valid=self.ego.at(point_times)
        xyz=frame[['x','y','z']].to_numpy(float)
        world=xyz@reference[:3,:3].T+reference[:3,3]
        laser=frame.laser_number.to_numpy(np.int64)
        other='down_lidar' if lower_ids_sensor=='up_lidar' else 'up_lidar'
        sensor=np.where((laser<32)[:,None,None],self.ego_from_sensor[lower_ids_sensor],self.ego_from_sensor[other])
        world_from_sensor=ego@sensor
        origin=world_from_sensor[:,:3,3]
        vectors=world-origin
        ranges=np.linalg.norm(vectors,axis=1)
        valid=pose_valid&reference_valid&(laser>=0)&(laser<64)&(ranges>0)
        return {'timestamp_ns':stamp,'point_timestamps_ns':point_times,'points_world_m':world,
            'origins_world_m':origin,'directions_world':vectors/np.maximum(ranges[:,None],1e-12),
            'observed_range_m':ranges,'valid_sensor_pose':valid,'laser_number':laser,
            'world_from_sensor':world_from_sensor,'reference_world_from_ego':reference,
            'lower_ids_sensor':lower_ids_sensor,'raw_points':len(frame)}

    def actor_coordinates(self,owner,sweep):
        if self.tracks is None: self.load_tracks()
        poses,known=self.tracks[owner]['poses'].at(sweep['point_timestamps_ns'])
        rotations=poses[:,:3,:3]
        points=np.einsum('nji,nj->ni',rotations,sweep['points_world_m']-poses[:,:3,3])
        origins=np.einsum('nji,nj->ni',rotations,sweep['origins_world_m']-poses[:,:3,3])
        directions=np.einsum('nji,nj->ni',rotations,sweep['directions_world'])
        return points,origins,directions,known&sweep['valid_sensor_pose']

    def camera(self,channel,filename,image_hw=(672,672)):
        """Keep the full undistorted view, letterbox and transform K exactly."""
        path=Path(filename)
        if not path.is_absolute(): path=self.root/path
        stamp=int(path.stem)
        ego,valid=self.ego.at(stamp)
        calibration=self.intrinsics.loc[channel]
        intrinsic=np.array([[calibration.fx_px,0,calibration.cx_px],
                            [0,calibration.fy_px,calibration.cy_px],[0,0,1]],float)
        height,width=image_hw
        with Image.open(path) as image:
            old_width,old_height=image.size
            scale=min(width/old_width,height/old_height)
            new_width=max(1,round(old_width*scale)); new_height=max(1,round(old_height*scale))
            left=(width-new_width)//2; top=(height-new_height)//2
            resized=image.convert('RGB').resize((new_width,new_height),Image.Resampling.LANCZOS)
            canvas=Image.new('RGB',(width,height)); canvas.paste(resized,(left,top))
            rgb=np.asarray(canvas,dtype=np.float32)/255
        affine=np.array([[new_width/old_width,0,left],[0,new_height/old_height,top],[0,0,1]],float)
        return {'image':rgb.transpose(2,0,1).copy(),'intrinsics':affine@intrinsic,
                'pixel_affine':affine,'original_intrinsics':intrinsic,
                'valid_image_rect_xyxy':[left,top,left+new_width,top+new_height],
                'world_from_camera':ego@self.ego_from_sensor[channel],
                'timestamp_ns':stamp,'pose_known':bool(valid),'camera_id':channel,'image_path':str(path)}
