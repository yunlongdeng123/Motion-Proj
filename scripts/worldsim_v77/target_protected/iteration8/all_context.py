"""精确保护对象清单覆盖实际训练短窗；不让四秒素材筛选遗漏一秒内真实邻车。"""
from pathlib import Path
import sys,os,copy
from collections import Counter
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
sys.path.insert(0,'/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected')
import numpy as np
from geometry_factory import read,dump
from asset_factory import ParkingGeometry,O,F
def main():
    geo=ParkingGeometry(F);manifest=read(F/'source_manifest.json');src={c['source_id']:c for c in manifest['clips']};added=[];reject=Counter()
    for sid,c in src.items():
        if not sid.startswith('N'):continue
        g=geo.prepare(sid)
        if not g['pass']:continue
        obs=geo.obstacles[sid];tokens={a['instance_token'] for a in obs[0] if a['category'] in ['vehicle.car','vehicle.truck','vehicle.bus.rigid','vehicle.bus.bendy']}
        known={a['instance_token'] for a in c['actors']}
        activekeys={int(i) for f in c['frames'] for i in [np.searchsorted(c['keyframe_timestamps'],f['timestamp'])-1,np.searchsorted(c['keyframe_timestamps'],f['timestamp'])]}
        context=geo.context[sid]['frames']
        for tok in sorted(tokens-known):
            arr=[next((a for a in row if a['instance_token']==tok),None) for row in obs]
            if any(a is None or a.get('interpolation_uncertain') for a in arr):reject['incomplete']+=1;continue
            ps=[a['_projection'] for a in arr]
            if any(p is None or p['box'][2]-p['box'][0]<72 or p['box'][3]-p['box'][1]<40 or min(p['box'][0],p['box'][1],1024-p['box'][2],576-p['box'][3])<8 for p in ps):reject['size_border']+=1;continue
            if any(not any(a['instance_token']==tok and int(a['visibility_token'])==4 for a in context[i]['annotations']) for i in activekeys):reject['visibility']+=1;continue
            c['actors'].append({'instance_token':tok,'category':arr[0]['category'],'context_extension':'actual one-second bracket observations; no extrapolation'})
            for f,a,p in zip(c['frames'],arr,ps):f['actors'].append({k:copy.deepcopy(a[k]) for k in ['instance_token','category','translation','rotation','size']}|{'projection':{'box_xyxy':p['box'].tolist(),'depth':p['center_depth']}})
            added.append({'source_id':sid,'instance_token':tok,'category':arr[0]['category']})
    if not (O/'source_manifest_before_all_context.json').exists():dump(O/'source_manifest_before_all_context.json',read(F/'source_manifest.json'))
    dump(F/'source_manifest.json',manifest);dump(O/'all_context_extension.json',{'added':added,'reject_counts':dict(reject),'scope':'well-observed real vehicle metadata at every exposure; SAM2 and independent final QA still required; no new RGB needed'})
    print('ALL_CONTEXT',len(added),'sources',len({r['source_id'] for r in added}),'reject',dict(reject),flush=True)
if __name__=='__main__':main()
