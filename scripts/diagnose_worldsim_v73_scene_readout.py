"""一次有解析真值的刚体组合：遮挡、背景、漏束与Actor移动，不跑训练回归。"""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v73.scene_readout import SurfaceBVH,compose_first

parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
vertices=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],np.float32)
faces=np.array([[0,1,2],[0,2,3]],np.uint32)
background=SurfaceBVH(vertices*3+[0,0,10],faces); actor=SurfaceBVH(vertices,faces)
origins=np.array([[.2,.1,0],[2,.1,0],[4,.1,0]],np.float32); directions=np.tile([0,0,1],(3,1)).astype(np.float32)
pose=np.eye(4); pose[:3,3]=[0,0,5]
pose[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]
rear=pose.copy(); rear[2,3]=7
bg=background.cast(origins,directions)
depth,owner,_=compose_first(bg,[(1,actor.cast(origins,directions,pose)),(2,actor.cast(origins,directions,rear))])
np.testing.assert_allclose(depth,[5,10,np.inf]); np.testing.assert_array_equal(owner,[1,0,-1])
pose[0,3]=2
moved,moved_owner,_=compose_first(bg,[(1,actor.cast(origins,directions,pose)),(2,actor.cast(origins,directions,rear))])
np.testing.assert_allclose(moved,[7,5,np.inf]); np.testing.assert_array_equal(moved_owner,[2,1,-1])
value={'status':'done','static_depth_m':[float(x) if np.isfinite(x) else None for x in depth],
    'static_owner':owner.tolist(),'edited_depth_m':[float(x) if np.isfinite(x) else None for x in moved],
    'edited_owner':moved_owner.tolist(),'meaning':'known analytic visibility and rigid coordinate conversion; not real scene quality or counterfactual truth'}
args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(value,indent=2)+'\n'); print(json.dumps(value))
