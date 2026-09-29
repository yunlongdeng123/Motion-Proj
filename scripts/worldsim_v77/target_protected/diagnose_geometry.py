"""原位几何正控制：真实actor的投影及地面位置，区分提案不足和坐标工程错误。"""
import argparse,math
from pathlib import Path
from collections import Counter
import numpy as np
from pyquaternion import Quaternion
from geometry_factory import Geometry,projection,footprint,ground_orientation,dump

def main(root):
    g=Geometry(root);rows=[]
    for sid in ['S008','S013','S024','S058','T003','T013','T015','T017','T027']:
        c=g.sources[sid];ground=g.prepare(sid)
        if not ground['pass']:continue
        details=[]
        for f in c['frames']:
            b=f['actors'][0];p=projection(b,f);saved=np.array(b['projection']['box_xyxy']);origerr=float(np.max(abs(p['box']-saved)))
            R=Quaternion(b['rotation']).rotation_matrix;yaw=math.atan2(R[1,0],R[0,0]);Rg=ground_orientation(yaw,ground['plane']);xy=np.array(b['translation'])[:2];bottom=np.r_[xy,np.dot(np.r_[xy,1],ground['plane'])];tr=bottom+Rg[:,2]*b['size'][2]/2
            a=b|{'translation':tr.tolist(),'rotation':Quaternion(matrix=Rg).elements.tolist()};pa=projection(a,f)
            diff=tr-np.array(b['translation']);ratio=(pa['box'][2:]-pa['box'][:2])/(saved[2:]-saved[:2]);others=[ob for ob in g.obstacles[sid][f['frame']] if ob['instance_token']!=b['instance_token']]
            details.append({'frame':f['frame'],'original_projection_error_px':origerr,'ground_center_shift_m':diff.tolist(),'ground_projection_scale_xy':ratio.tolist(),
                            'original_footprint_in_drivable':bool(g.road.covers(footprint(b))),
                            'original_clearance_to_other_footprints_m':min(footprint(b).distance(ob['_foot']) for ob in others)})
        failures=Counter()
        for z in [-6,-4,-2,0,2,4,5.5,7,9]:
            for x in [-6,-3,-1.5,0,1.5,3,6]:
                a,why=g.trajectory(sid,z,x,sid)
                failures[why or 'geometry_pass']+=1
        row={'source_id':sid,'frames':details,'same_donor_geometry_control':dict(failures)};rows.append(row)
        print(sid,'projection_max',max(r['original_projection_error_px'] for r in details),'ground_z',[round(min(r['ground_center_shift_m'][2] for r in details),2),round(max(r['ground_center_shift_m'][2] for r in details),2)],'onroad',sum(r['original_footprint_in_drivable'] for r in details),dict(failures),flush=True)
    dump(root/'geometry_positive_controls.json',{'scope':'same donor controls are diagnostic only, never training samples','clips':rows})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
