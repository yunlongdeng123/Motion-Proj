"""用独立真实扫描排查已知静态占据；稀疏零点不能证明空间空。"""
from pathlib import Path
import sys,os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from pyquaternion import Quaternion
sys.path.insert(0,str(Path(__file__).parent))
from asset_factory import O,F,ParkingGeometry,read,dump
from geometry_factory import transform

def clouds(geo,sid):
    ctx=geo.context[sid]['frames'];rows=[]
    for k in [0,3,6]:
        f=ctx[k];d=f['sensors']['LIDAR_TOP'];raw=np.fromfile(F/'rgb'/d['filename'],dtype=np.float32).reshape(-1,5)[:,:3]
        cal=d['calibrated_sensor'];ego=d['ego_pose'];m=transform(ego['translation'],ego['rotation'])@transform(cal['translation'],cal['rotation']);pts=raw@m[:3,:3].T+m[:3,3]
        for a in f['annotations']:
            loc=(pts-np.array(a['translation']))@Quaternion(a['rotation']).rotation_matrix;w,l,h=a['size'];pts=pts[~np.all(abs(loc)<np.array([l/2+.2,w/2+.2,h/2+.2]),axis=1)]
        rows.append({'key':k,'timestamp':d.get('timestamp',None),'points':pts})
    return rows

def main():
    geo=ParkingGeometry(F);cache={};cases=[]
    for c in read(O/'data_review/synthetic_manifest.json')['clips']:
        sid=c['source_id'];geo.prepare(sid)
        if sid not in cache:cache[sid]=clouds(geo,sid)
        plane=np.array(c['ground']['plane']);stats=[]
        for pose in c['frames']:
            a=pose['actor'];R=Quaternion(a['rotation']).rotation_matrix;size=np.array([a['size'][1],a['size'][0],a['size'][2]]);hit=[]
            for cloud in cache[sid]:
                pts=cloud['points'];local=(pts-np.array(a['translation']))@R
                height=pts[:,2]-np.c_[pts[:,:2],np.ones(len(pts))]@plane
                keep=(abs(local[:,0])<size[0]/2)&(abs(local[:,1])<size[1]/2)&(height>.15)&(height<size[2]-.1)
                voxels=np.unique(np.floor(pts[keep]/.2).astype('int32'),axis=0)
                hit.append({'key':cloud['key'],'points':int(keep.sum()),'distinct_20cm_voxels':len(voxels)})
            positive=sum(r['distinct_20cm_voxels']>=3 for r in hit)==3
            stats.append({'frame':pose['frame'],'scan_counts':hit,'three_scan_static_positive':positive})
        case={'case_id':c['case_id'],'source_id':sid,'scene':c['scene'],'frames':stats,'positive_frames':sum(r['three_scan_static_positive'] for r in stats),'static_input_reject':any(r['three_scan_static_positive'] for r in stats),'limits':'GT boxes removed per actual scan; >=3 unique20cm occupied voxels in all3 independent registered scans above fitted ground. Positive is conservative bbox occupancy; zero does not certify empty or full road legality.'}
        cases.append(case);dump(O/'static_occupancy.json',{'stage':'running','cases':cases});print('STATIC',c['case_id'],case['positive_frames'],flush=True)
    dump(O/'static_occupancy.json',{'stage':'complete','cases':cases,'positive_cases':sum(c['static_input_reject'] for c in cases),'empty_scan_does_not_certify_empty_space':True,'uniform_rule_applied_to_all_cases':True})
    tech=read(O/'technical_checks.json');byid={c['case_id']:c for c in cases}
    assert tech['stage']=='complete' and len(tech['cases'])==len(cases)
    for c in tech['cases']:
        static=byid[c['case_id']];c['static_occupancy']=static;c['technical_pass']=c['technical_pass'] and not static['static_input_reject']
    tech['all_pass']=all(c['technical_pass'] for c in tech['cases']);tech['static_scan_check_complete']=True;dump(O/'technical_checks.json',tech)
if __name__=='__main__':main()
