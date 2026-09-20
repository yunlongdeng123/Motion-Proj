"""普通图像框IoU关联控制；保留既有米制Kalman与所有检测，不称完整SORT。"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from rgb_idm_policy import MotionTracks


class ImageIoUMotionTracks(MotionTracks):
    """只替换关联依据，IoU门槛0.3取自SORT官方默认；不扫描参数。"""
    def associate(self, detections):
        if not self.tracks or not detections: return {}
        a = np.array([t['last_box'] for t in self.tracks])[:,None,:]
        b = np.array([d['box'] for d in detections])[None,:,:]
        size = np.maximum(0.,np.minimum(a[...,2:],b[...,2:])-np.maximum(a[...,:2],b[...,:2]))
        intersection = size.prod(-1)
        area_a = (a[...,2:]-a[...,:2]).prod(-1); area_b = (b[...,2:]-b[...,:2]).prod(-1)
        overlaps = intersection/np.maximum(area_a+area_b-intersection,1e-12)
        valid = overlaps >= .3
        # 与旧分配一样先门控，避免无效配对挤掉有效匹配。
        cost = np.where(valid,1-overlaps,float(max(overlaps.shape)+1))
        ti,di = linear_sum_assignment(cost)
        return {int(d):self.tracks[int(t)] for t,d in zip(ti,di) if valid[t,d]}

    def remember_detection(self, track, detection):
        track['last_box'] = list(detection['box'])
