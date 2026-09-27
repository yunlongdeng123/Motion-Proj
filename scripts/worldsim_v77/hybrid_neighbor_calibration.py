"""r13：固定的actor局部尺度控制；训练/留出LiDAR锚点分开，不放宽来源准入。"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import hybrid_neighbor_evidence as runner
from hybrid_neighbor_evidence import *

def build(spec,inst,out):
 records=[]
 def calibrate(d,fr,b,cam,c,k,lid):
  cp=transform(lid,np.linalg.inv(c));q=cp@k.T;uv=np.rint(q[:,:2]/np.maximum(q[:,2:],1e-6)).astype(int)
  ok=(cp[:,2]>.5)&(uv[:,0]>=0)&(uv[:,0]<d.shape[1])&(uv[:,1]>=0)&(uv[:,1]<d.shape[0])
  for other in fr['all_boxes']:
   if other['actor_id']==ACTOR:continue
   ids=np.flatnonzero(ok)
   if len(ids):ok[ids]&=~segment_box_occlusion(c[:3,3],lid[ids],np.array(other['pose']),np.array(other['size_lwh'])+.1)
  ids=np.flatnonzero(ok)
  # 图像横坐标/纵坐标排序后交替留出；同一点不同时拟合与验证。
  ids=ids[np.lexsort((uv[ids,1],uv[ids,0]))];fit=ids[::2];test=ids[1::2]
  rec=dict(frame=int(fr['frame']) if 'frame' in fr else None,camera=cam,fit_anchors=len(fit),heldout_anchors=len(test),admitted=False)
  # fr未必带frame；source顺序由输入相机位姿和源索引再次记录。
  rec['camera_center']=c[:3,3].tolist()
  if min(len(fit),len(test))<8:
   rec['reason']='insufficient_separate_anchors';records.append(rec);return None
  pred=d[uv[fit,1],uv[fit,0]];scale=float(np.median(cp[fit,2]/pred))
  raw=d[uv[test,1],uv[test,0]];err=np.abs(scale*raw-cp[test,2]);old=np.abs(raw-cp[test,2])
  rec.update(scale=scale,heldout_raw_median_m=float(np.median(old)),heldout_corrected_median_m=float(np.median(err)),heldout_corrected_q90_m=float(np.quantile(err,.9)))
  # 预登记的门槛，在看新投影前固定；不搜索阈值。
  if not (.5<scale<2 and np.median(err)<=.25 and np.quantile(err,.9)<=.5):
   rec['reason']='heldout_geometry_rejected';records.append(rec);return None
  rec.update(admitted=True,reason='heldout_geometry_supported');records.append(rec)
  return d*scale
 sources,index=runner.prepare_sources(spec,inst,out,depth_transform=calibrate)
 dump(out/'calibration.json',dict(records=records,scope='GT/LiDAR-assisted local scale control; heldout sparse depth only, not RGB/identity/occlusion proof',human_verdict=None))
 return sources,index

if __name__=='__main__':
 runner.main('r13',build,dict(hypothesis='r12 evidence may be rejected by object-local metric depth bias; holdout-tested one-scale correction can recover some valid points',
    change='Local median LiDAR/Omega depth ratio only, all visibility/LiDAR20cm/two-source/erosion rules unchanged',
    local_calibration=dict(min_fit_anchors=8,min_heldout_anchors=8,split='sorted image x/y alternating fit/test',estimator='median metric-depth ratio',scale_range=[.5,2],heldout_median_max_m=.25,heldout_q90_max_m=.5),
    inputs='Same as r12; additional actor-local GT/LiDAR calibration, no network training',human_verdict=None))
