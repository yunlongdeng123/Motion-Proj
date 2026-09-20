"""用已知相机标定审计点图投影；PnP只作普通坐标控制，不使用目标真值。"""
import argparse
import json
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from prepare_natural import OUT

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,default=OUT)
    OUT=parser.parse_args().run_dir
    p=json.loads((OUT/'protocol.json').read_text())
    native=np.load(OUT/'native_outputs.npz')['points'][0,0]/.1
    rows=[];gauges=[]
    yy,xx=np.mgrid[4:512:8,4:512:8];uv=np.stack([xx,yy],-1).reshape(-1,2).astype(float)
    for i,v in enumerate(p['views']):
        point=native[i,yy,xx].reshape(-1,3).astype(float)
        good=(np.linalg.norm(point,axis=-1)<80)&(np.linalg.norm(point,axis=-1)>2)&(uv[:,1]>=v['pad_top']+4)&(uv[:,1]<508-v['pad_top'])
        point=point[good];pixel=uv[good]
        K=np.array(v['K_network']);cv2.setRNGSeed(7501)
        ok,r,t,inliers=cv2.solvePnPRansac(point,pixel,K,None,iterationsCount=200,reprojectionError=5,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            rows.append({'camera':v['camera'],'pnp_pass':False});continue
        r,t=cv2.solvePnPRefineLM(point[inliers[:,0]],pixel[inliers[:,0]],K,None,r,t)
        pred,_=cv2.projectPoints(point,r,t,K,None);err=np.linalg.norm(pred[:,0]-pixel,axis=-1)
        E=np.eye(4);E[:3,:3]=cv2.Rodrigues(r)[0];E[:3,3]=t[:,0]
        gauge=np.array(v['camera_world'])@E;gauges.append(gauge)
        rows.append({'camera':v['camera'],'pnp_pass':True,'n':len(point),'inliers':len(inliers),
                     'median_error_all_px':float(np.median(err)),
                     'median_error_inliers_px':float(np.median(err[inliers[:,0]])),
                     'world_from_native_via_camera':gauge.tolist()})
    if len(gauges)==7:
        meanR=Rotation.from_matrix(np.array(gauges)[:,:3,:3]).mean().as_matrix()
        meant=np.median(np.array(gauges)[:,:3,3],axis=0)
        for row,gauge in zip(rows,gauges):
            row['gauge_rotation_deviation_deg']=float(np.rad2deg(Rotation.from_matrix(meanR.T@gauge[:3,:3]).magnitude()))
            row['gauge_translation_deviation_m']=float(np.linalg.norm(gauge[:3,3]-meant))
    result={'status':'complete','kind':'projection_and_common_gauge_control','rows':rows,
            'human_verdict':None,'gt_target_translation_used':False,
            'interpretation':'原生点图坐标无法直接挂到已知rig时，先检查是否一个普通共同刚体变换即可解释；不按该结果重排输入或改模型'}
    path=OUT/'geometry_audit.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
