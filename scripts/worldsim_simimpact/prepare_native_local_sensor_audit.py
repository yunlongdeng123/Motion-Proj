"""单一稳定漏检对象的有限传感器定位；不冒称重建资产修复。"""
import json,shutil,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from pyquaternion import Quaternion
D=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-DETECTOR-01/20260915-r1')
O=D/'local_excavator_inputs_r1'
if O.exists():raise RuntimeError(f'Preserve {O}')
O.mkdir();shutil.copy2(__file__,O/'source_snapshot.py')
instance='acb7920a70b44a84b1a61ce6460483cb'
objects=json.loads((D/'object_audit_r2/objects.json').read_text());realrows=json.loads((D/'real_scene0004_r1/summary.json').read_text())
reg={'task_id':'WS-SIM-NATIVE-DETECTOR-LOCAL-01','run_id':O.name,'scenes':['scene-0004'],'seed':20260915,'instance':instance,
     'selection':'Single construction vehicle: real8/8, raw0/8, median1/8 at score0.5, class-aware IoU0.5; already exposed eight-query discovery window.',
     'scope':'Finite local sensor audit, not geometry-asset repair. GT oriented box expanded by0.25m, chosen once, includes its neighborhood; cannot label every changed point as object surface.',
     'information':'Evaluator-held GT boxes and real ten-sweep scans are oracle extra inputs. Actual geometry model/checkpoint unchanged.',
     'conditions':[f'{r}_{c}' for r in ['raw','median'] for c in ['local_real','local_xyz','local_intensity']],
     'interventions':{'local_real':'Replace all predicted points inside the region with real points in the same region.',
                      'local_xyz':'For predicted points inside region, copy nearest real position from the same sweep, retaining each predicted intensity/age/count. Many-to-one correspondences are possible; not exact surface restoration.',
                      'local_intensity':'At fixed predicted positions, copy intensity from nearest real point of the same sweep, retaining geometry/age/count.'},
     'stop_rule':'Exactly three local operations, two readouts, eight existing query times. Full local replacement tests localization; intensity recovery provides an alternative explanation. Only persistent geometry-specific recovery warrants actual asset intervention; no smaller-box or threshold sweep.',
     'failure_ledger_refs':['V74-H2-F18'],'failure_ledger_delta':'pending','human_verdict':None,'prepared':False,'start_unix':time.time()}
(O/'registration.json').write_text(json.dumps(reg,indent=2))
frames=[];stats=[]
for i in range(8):
    obj=next(o for o in objects if o['index']==i and o['instance']==instance)
    center=np.array(obj['center_lidar']);rot=Quaternion(obj['rotation']).rotation_matrix;half=np.array(obj['wlh'])[[1,0,2]]/2+.25
    def mask(p):return (abs((p[:,:3]-center)@rot)<half).all(1)
    real=np.load(D/f'real_scene0004_r1/input_{i:02d}.npz')['points'];real_local=real[mask(real)];dest=O/f'frame_{i:02d}';dest.mkdir();paths={}
    for readout in ['raw','median']:
        pred=np.load(D/f'native_scans_scene0004_step030000_r1/{readout}_{i:02d}.npz')['points'];inside=mask(pred);inds=np.flatnonzero(inside)
        xyz=pred.copy();intensity=pred.copy();distances=[]
        for age in np.unique(pred[inside,4]):
            selected=inds[np.isclose(pred[inds,4],age,atol=1e-7,rtol=0)];reference=real[np.isclose(real[:,4],age,atol=1e-7,rtol=0)]
            if not len(reference):raise RuntimeError(f'Missing same-time reference {i} {age}')
            dist,nn=cKDTree(reference[:,:3]).query(pred[selected,:3]);xyz[selected,:3]=reference[nn,:3];intensity[selected,3]=reference[nn,3];distances.extend(dist.tolist())
        variants={'local_real':np.r_[pred[~inside],real_local],'local_xyz':xyz,'local_intensity':intensity}
        for kind,p in variants.items():
            path=dest/f'{readout}_{kind}.npz';np.savez_compressed(path,points=p.astype(np.float32));paths[f'{readout}_{kind}']=str(path)
        stats.append({'index':i,'readout':readout,'pred_changed_points':len(inds),'real_replacement_points':len(real_local),'position_move_quantiles_m':np.quantile(distances,[.5,.9,1]).tolist(),'outside_region_unchanged':True})
    frames.append({'scene':'scene-0004','sample_token':realrows[i]['sample_token'],'points':paths})
reg.update(prepared=True,expected_detector_forwards=48,intervention_statistics=stats,end_unix=time.time())
(O/'registration.json').write_text(json.dumps(reg,indent=2));(O/'input_manifest.json').write_text(json.dumps({'registration':reg,'frames':frames},indent=2));print('LOCAL_SENSOR_INPUTS_PREPARED',len(frames),flush=True)
