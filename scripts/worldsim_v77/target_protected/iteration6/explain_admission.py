"""记录原精确关卡拒绝时的实际保护比例；只观察，不更改阈值或重选位姿。"""
from pathlib import Path
import argparse,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from geometry_factory import Geometry,read,dump
from iteration5.planning import inputs,silhouettes
from build_pairs import evaluate

def main(parent,root):
    sources,donors,donor_root,pairs,_=inputs(parent);geo=Geometry(parent/'native10_factory');rows=[]
    for p in pairs:
        sid=p['source_id'];d=donors[p['donor_source_id']];geo.sources[d['source_id']]=d
        for f in d['frames']:f['_w2c']=np.linalg.inv(f['camera_to_world'])
        assert geo.prepare(sid)['pass'];traj,why=geo.trajectory(sid,p['offset_longitudinal_m'],p['offset_lateral_m'],d['source_id'],p['placement_mode'],0);assert traj
        masks,contacts,_=silhouettes(p,d,donor_root)
        protected={tok:[np.asarray(__import__('PIL.Image',fromlist=['Image']).open(parent/'segmented_receivers'/(sid+'_'+tok[:8])/'sam2_raw'/f'{i:05}.png'))>0 for i in range(10)] for tok in p['required_protected_instances']}
        snapshot={}
        def observe(frame,event,arg):
            if frame.f_code==evaluate.__code__ and event=='return':
                for k in ('active','overlaps','hole_overlaps','contact_errors','unverified_hits'):
                    if k in frame.f_locals:snapshot[k]=frame.f_locals[k]
            return observe
        sys.settrace(observe)
        try:quality,reason=evaluate(geo,traj,masks,contacts,protected,ego_bottom_guard_px=64)
        finally:sys.settrace(None)
        rows.append({'case_id':sid,'exact_gate_pass':quality is not None,'reason':reason,'actual_mask_stats':snapshot,'thresholds':{'single_silhouette_overlap_range':[.3,.8],'consecutive_frames':10,'single_max_silhouette_overlap':.85,'single_max_hole_overlap':.85},'human_verdict':None})
    dump(root/'exact_mask_diagnostics.json',{'cases':rows,'classification':'read-only original evaluate locals at return; thresholds unchanged; no new proposal search'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.parent,a.root)
