"""检查磁盘上的实际训练输入；不能仅复用提案的pass字段。"""
from pathlib import Path
import sys,json,os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).parent))
from asset_factory import O,F,ParkingGeometry,trajectory,silhouette,exact,source_masks,read,dump
from iteration5.planning import normalized_masks

def pixel_gates(y,x,influence,hole,protected):
    assert y.shape==x.shape==(576,1024,3)
    assert hole.shape==influence.shape==(576,1024)
    assert influence.any() and hole.any()
    assert not np.any(influence&~hole),'A influence漏出模型洞'
    assert not np.any((x!=y).any(-1)&~hole),'synthetic RGB影响漏出模型洞'
    assert not hole[512:].any(),'hole压到ego保守禁入带'
    masked_x=x.copy();masked_y=y.copy();masked_x[hole]=0;masked_y[hole]=0
    assert np.array_equal(masked_x,masked_y),'条件中仍残留合成RGB'
    for tok,m in protected.items():
        assert m.shape==hole.shape and m.any(),('empty protected mask',tok)
        assert (hole&m).sum()/m.sum()<=.85,('protected evidence exhausted',tok)
    return {'synthetic_RGB_leak':0,'influence_outside_hole':0,'condition_X_equals_masked_Y':True,'ego_hole_pixels':0}

def main():
    geo=ParkingGeometry(F);rows=[]
    for c in read(O/'data_review/synthetic_manifest.json')['clips']:
        folder=Path(c['folder']);sid=c['source_id'];src=geo.sources[sid];g=geo.prepare(sid)
        assert g['pass'];size=c['frames'][0]['actor']['size']
        tr,why=trajectory(geo,sid,c['offset_longitudinal_m'],c['offset_lateral_m'],size)
        assert tr is not None,why
        for actual,stored in zip(tr['frames'],c['frames']):
            for key in ['translation','size','rotation']:
                assert np.allclose(actual['actor'][key],stored['actor'][key],atol=1e-6)
        mesh=dict(np.load(O/'assets'/(c['asset']+'.npz')))
        aa=[silhouette(mesh['vertices'],mesh['faces'],p['actor'],f) for p,f in zip(tr['frames'],src['frames'])]
        pm={}
        for j,a in enumerate(src['actors']):
            tok=a['instance_token'];jid=sid if j==0 else sid+'_'+tok[:8]
            pf=F/('segmented' if j==0 else 'segmented_secondary')/jid/'sam2_raw'
            if not pf.exists():continue
            mm=source_masks(F,sid,None if j==0 else jid)
            stats=normalized_masks(mm,[next(a for a in f['actors'] if a['instance_token']==tok)['projection']['box_xyxy'] for f in src['frames']])
            if all(s['pixels']>0 and s['normalized_iou']>=.8 and .85<=s['normalized_area_ratio']<=1.15 for s in stats):pm[tok]=mm
        q,why=exact(geo,tr,aa,pm);assert q is not None,why;assert q['type']==c['type']
        pixels=[]
        for i,f in enumerate(src['frames']):
            load=lambda r,rgb:np.asarray(Image.open(folder/r/f'{i:03}.png').convert('RGB' if rgb else 'L'))
            y=load('Y',True);x=load('X',True);h=load('model_hole',False)>0;infl=load('influence',False)>0
            with Image.open(F/'rgb'/f['filename']) as im:
                assert np.array_equal(y,np.asarray(im.convert('RGB').resize((1024,576),Image.Resampling.LANCZOS))),'Y不是真实RGB'
            stored={tok:np.asarray(Image.open(folder/'protected'/f'{i:03}_{tok}.png'))>0 for tok in pm}
            assert all(np.array_equal(stored[tok],mm[i]) for tok,mm in pm.items())
            pixels.append(dict(frame=i,**pixel_gates(y,x,infl,h,stored)))
        row={'case_id':c['case_id'],'type':c['type'],'scene':c['scene'],'frames_checked':10,'technical_pass':True,'pixels':pixels,'ground':tr['ground'],'min_GT_clearance_m':tr['min_GT_clearance_m'],'max_ground_support_distance_m':tr['max_ground_support_distance_m'],'silhouette_stats':q['silhouette_stats'],'depth_and_type_revalidated':True,'independent_visual_QA':'pending; numeric mask consistency is not identity certification','human_verdict':None}
        rows.append(row);dump(O/'technical_checks.json',{'stage':'running','cases':rows});print('VALIDATED',c['case_id'],flush=True)
    dump(O/'technical_checks.json',{'stage':'complete','cases':rows,'checked_frames':len(rows)*10,'all_pass':all(r['technical_pass'] for r in rows),'limits':'map/GT/LiDAR-assisted input checks; one-second windows; static objects behind A allowed, dynamic unreviewed envelopes rejected; technical pass is not model success'})
if __name__=='__main__':main()
