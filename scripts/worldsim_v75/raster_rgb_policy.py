"""普通策略的地形适配；预热只接起点前真实RGB，与生成条件不共享GTactor。"""
import numpy as np
from raster_ground import RasterGround
from rgb_idm_policy import RGBIDMPolicy


class RasterRGBIDMPolicy(RGBIDMPolicy):
    def __init__(self, base, detector=None):
        self.terrain=RasterGround(base)
        trajectory=np.load(base/'trajectory.npz')
        extrinsic=np.linalg.inv(trajectory['ego_world'][0])@trajectory['camera_world'][0]
        super().__init__(trajectory['ego_world'][:,:3,3],trajectory['K'],extrinsic,[0,0,0],detector)

    def contact(self,u,v,camera): return self.terrain.contact(u,v,camera,self.K)

    def tracker_snapshot(self):
        return {'previous_time':self.tracker.previous_time,'serial':self.tracker.serial,
                'tracks':[{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in t.items()} for t in self.tracker.tracks]}

    def restore_tracker(self,snapshot):
        self.tracker.previous_time=snapshot['previous_time']; self.tracker.serial=snapshot['serial']
        self.tracker.tracks=[{k:np.array(v) if k in ['state','cov'] else v for k,v in t.items()} for t in snapshot['tracks']]
