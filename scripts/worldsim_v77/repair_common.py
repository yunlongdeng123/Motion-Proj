"""三场景DELETE入口修复；每项来源可追溯，旧结果不改写。"""
import json,pathlib,sys
import numpy as np,cv2
ROOT=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-DELETE-REPAIR-20260927/r1')
REPO=pathlib.Path('/root/autodl-tmp/motion_proj_v77')
FULL=ROOT.parents[1]/'WS-V77-DELETE-FULL-20260927/r1'
VIDEO=ROOT.parents[1]/'WS-V77-VIDEO-REVIEW-20260926/r1'
P0=ROOT.parents[1]/'WS-V77-P0-24ACTOR-20260926/r1'
sys.path.insert(0,str(REPO/'scripts/worldsim_v77'))
from geometry import transform,resized_intrinsics,project_bbox,unproject,box_mask
from video_review import scene_frame
def dump(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def read(p):return json.loads(pathlib.Path(p).read_text())
def hull_mask(b,c2w,k,hw=(576,1024),pad=0,ground_extend=0):
 pose=np.array(b['pose']);size=np.array(b['size_lwh']);size[2]+=ground_extend;pose[2,3]-=ground_extend/2
 signs=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
 cp=transform(transform(signs*size/2,pose),np.linalg.inv(c2w));m=np.zeros(hw,np.uint8)
 if cp[:,2].min()<=.2:return m.astype(bool)
 q=cp@k.T;uv=np.rint(q[:,:2]/q[:,2:]).astype('int32');cv2.fillConvexPoly(m,cv2.convexHull(uv),1)
 if pad:m=cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*pad+1,2*pad+1)))
 return m>0
def camera(fr,c,hw=(576,1024)):
 v=fr['views'][c];return np.array(v['c2w']),resized_intrinsics(v['intrinsics'],v['original_wh'],hw)
def largest(m):
 n,l,st,_=cv2.connectedComponentsWithStats(m.astype('uint8'),8)
 return l==(1+int(np.argmax(st[1:,cv2.CC_STAT_AREA]))) if n>1 else m
