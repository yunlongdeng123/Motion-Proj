"""Independent saved-array components/foreground audit; no model or GPU.

Use alongside (not instead of) the unchanged4560 fixed-domain score. Supports
H4 and future C-material checkpoints; --initial-case adds C0 frozen-geometry
checks. Intended silver/background changes are diagnostics, not blanket vetoes.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import hashlib
import json
from collections import deque
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

GROUPS=('asset_C2','native_a37','oldroad','native_static','semantic_a5','unused_a18','unused_a26','newroad')
COLORS=np.array([[224,72,72],[142,70,190],[244,161,37],[43,128,205],[36,164,81],[170,170,170],[110,110,110],[21,200,191]],np.uint8)
SILVER={128:(210,202,302,242),137:(207,207,274,227)}
LUMA=np.array([.2126,.7152,.0722],np.float64)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def snap(p):
    s=p.stat();return {'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':sha(p)}
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def groups(d):
    g=d['group_alpha_T'];assert g.shape in ((7,320,480),(8,320,480)) and np.isfinite(g).all()
    return np.concatenate([g,np.zeros_like(g[:1])],0) if len(g)==7 else g
def stat(v,m):
    x=np.asarray(v)[m].astype(np.float64);return {'pixels':int(m.sum()),'mean':float(x.mean()) if x.size else None,'max':float(x.max()) if x.size else None}
def component_stats(g):
    return {'alphaT_mean':dict(zip(GROUPS,g.astype(np.float64).mean((1,2)).tolist())),
            'dominant_group_pixels':dict(zip(GROUPS,np.bincount(g.argmax(0).ravel(),minlength=8).tolist()))}


def dark_components(dy,threshold,masks):
    binary=dy<threshold;seen=np.zeros_like(binary);rows=[]
    for sy,sx in zip(*np.nonzero(binary)):
        if seen[sy,sx]:continue
        q=deque([(int(sy),int(sx))]);seen[sy,sx]=True;coords=[]
        while q:
            y,x=q.popleft();coords.append((y,x))
            for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                if 0<=yy<320 and 0<=xx<480 and binary[yy,xx] and not seen[yy,xx]:seen[yy,xx]=True;q.append((yy,xx))
        c=np.asarray(coords);y,x=c.T
        rows.append({'pixels':len(coords),'bbox_xyxy':[int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1],
          'luma_delta_mean':float(dy[y,x].mean()),'luma_delta_min':float(dy[y,x].min()),
          'fixed_domain_intersections':{name:int(mask[y,x].sum()) for name,mask in masks.items()}})
    rows.sort(key=lambda r:(-r['pixels'],r['bbox_xyxy']))
    return {'diagnostic_threshold':threshold,'connectivity':4,'pixels':int(binary.sum()),'components':len(rows),'largest20':rows[:20],
            'role':'Post-render diagnostic, not a new score domain or quality threshold'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--pairs',nargs='+',help='Actual available RAW:state pairs; partial results remain partial');p.add_argument('--world-mask-root',type=Path);p.add_argument('--initial-case',type=Path);p.add_argument('--obs-root',type=Path);p.add_argument('--method-contract',type=Path,required=True)
    p.add_argument('--intentional-effective-alpha-change',action='store_true',help='Registered opacity handover; global render-alpha change is reported, not a colour-only invariant')
    p.add_argument('--candidate-training-receipt',type=Path);p.add_argument('--raws',nargs='+',type=int,default=[94,105,128,137]);p.add_argument('--case-label',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
    pairs=[(int(v.split(':')[0]),v.split(':')[1]) for v in a.pairs] if a.pairs else [(r,s) for r in a.raws for s in ['keep','delete']]
    assert len(pairs)==len(set(pairs)) and all(s in ('keep','delete') for r,s in pairs)
    complete=set(pairs)=={(r,s) for r in [94,105,128,137] for s in ['keep','delete']}
    paths={Path(__file__),a.method_contract}
    if a.candidate_training_receipt:paths.add(a.candidate_training_receipt)
    for raw,state in pairs:
        if a.obs_root:paths.add(a.obs_root/f'{raw:03d}.npz')
        for case in [a.baseline,a.candidate]+([a.initial_case] if a.initial_case else []):
            paths.add(case/f'frames/{raw:03d}/{state}_float.npz')
    if a.world_mask_root:
        paths.add(Path(__file__).with_name('registered_seam_pairs.py'))
        paths.add(a.world_mask_root/'registry.json');paths.add(a.world_mask_root/'primary_raw128/fixed_masks_raw128.npz')
        paths.update(a.world_mask_root/f'{r:03d}.npz' for r in [94,105,128,137])
    before=[snap(path) for path in sorted(paths)]
    report={'schema':'SAVED_ROAD_COMPONENTS_CPU_V1','case':a.case_label,'status':'COMPLETED_COMPONENT_DIAGNOSTIC_NOT_QUALITY_ACCEPTANCE' if complete else 'PARTIAL_AVAILABLE_STATES_ONLY_NOT_FOUR_FRAME_COMPLETION',
      'actual_pairs':pairs,'complete_four_frame_pilot':complete,'expected_full_pilot_states':8,
      'group_order':GROUPS,'method_contract_sha256':sha(a.method_contract),'raws':a.raws,'records':[],'figures':[],
      'global_render_geometry_equality_required':not a.intentional_effective_alpha_change,
      'intrinsic_parameter_freeze':'Must be independently checked from actual immutable source/receipt; effective-opacity changes are not optimizer updates.',
      'actual_new_GPU_forwards':0,'actual_new_backwards':0,'actual_new_optimizer_updates':0,
      'candidate_training_receipt':json.loads(a.candidate_training_receipt.read_text()) if a.candidate_training_receipt else None,
      'quality_accepted':False,'human_verdict':None,
      'limits':['Model foreground domains are frozen baseline alphaT>0, never candidate masks.',
        'Source-component alphaT is not a current semantic truth or an RGB decomposition.',
        'A material gain cannot by itself repair fixed geometry/C2-alpha loss.',
        'Darkening components and silver rectangles diagnose intended changes versus artifacts; they do not replace4560 or actual image review.']}
    for raw,state in pairs:
        obs=load(a.obs_root/f'{raw:03d}.npz') if a.obs_root else None
        b=load(a.baseline/f'frames/{raw:03d}/{state}_float.npz');c=load(a.candidate/f'frames/{raw:03d}/{state}_float.npz')
        bg,cg=groups(b),groups(c);br,cr=b['rgb'],c['rgb'];assert br.shape==cr.shape==(3,320,480)
        fixed={'C2_model_positive':bg[0]>0,'semantic5_model_positive':bg[4]>0}
        if state=='keep':fixed['KEEP37_model_positive']=bg[1]>0
        foreground_union=np.zeros((320,480),bool)
        for m in fixed.values():foreground_union|=m
        masks={**fixed}
        if obs is not None:masks.update(observed_full37=obs['actor_labels']==5,observed_any_actor=obs['actor_labels']>0,road=obs['road'],road_edit=obs['road_edit'])
        row={'raw':raw,'state':state,'foreground':{},'source_group_partition_max_abs':float(abs(cg.sum(0)-c['alpha']).max()),
             'global_geometry_vs_initial':None,'silver_ROI':None}
        row['foreground_union_newroad_mass']={'pixels':int(foreground_union.sum()),
            'baseline_sum':float(bg[7][foreground_union].astype(np.float64).sum()),
            'candidate_sum':float(cg[7][foreground_union].astype(np.float64).sum()),
            'candidate_max':float(cg[7][foreground_union].max()) if foreground_union.any() else None,
            'candidate_positive_pixels':int((cg[7][foreground_union]>0).sum()),
            'zero_mass':bool(not (cg[7][foreground_union]>0).any()),'role':'Baseline model-positive union; no candidate-based domain selection'}
        if obs is not None:
            a5=obs['actor_labels']==1;y,x=np.nonzero(a5)
            a5upper=a5&(np.arange(320)[:,None]<=y.min()+.25*(y.max()-y.min())) if len(y) else np.zeros_like(a5)
            a6,a37=obs['visible'];guard=~obs['road']&~obs['road_edit']&(obs['protect']|a5|a6|a37)
            original_masks={'a5_RGB':a5,'a5_upper_RGB':a5upper,'a6_RGB':a6,
                'protected_nonroad_RGB':obs['protect']&~a6&~a37&~obs['road']&~obs['road_edit'],'original_road_RGB':obs['road']}
            if state=='keep':original_masks['a37_RGB']=a37
            gt=obs['rgb'].transpose(2,0,1);base_error=abs(br-gt).mean(0);candidate_error=abs(cr-gt).mean(0)
            row['original_fixed_domain_partial_metrics']={}
            for name,m in original_masks.items():
                bs,cs=stat(base_error,m),stat(candidate_error,m);delta=cs['mean']-bs['mean'] if m.any() else None
                row['original_fixed_domain_partial_metrics'][name]={'baseline':bs,'candidate':cs,'MAE_delta':delta,'original_threshold':1e-6,'passed':None if delta is None else delta<=1e-6}
            m=obs['depth_valid'][0];bs,cs=stat(abs(b['depth_m']-obs['depth_m'][0]),m),stat(abs(c['depth_m']-obs['depth_m'][0]),m)
            delta=cs['mean']-bs['mean'] if m.any() else None
            row['original_fixed_domain_partial_metrics']['a6_depth_m']={'baseline':bs,'candidate':cs,'MAE_delta':delta,'original_threshold_m':1e-5,'passed':None if delta is None else delta<=1e-5}
            row['original_nonroad_guard_partial']={'pixels':int(guard.sum()),'alpha_pixel_max_abs':stat(abs(c['alpha']-b['alpha']),guard)['max'],
                'depth_pixel_max_abs_m':stat(abs(c['depth_m']-b['depth_m']),guard)['max']}
            if a.world_mask_root:
                from registered_seam_pairs import seam_pairs
                row['original_seam_pair_partial']=[]
                for pair in seam_pairs(raw,a.world_mask_root):
                    l,r=pair['minus_mask'],pair['plus_mask'];bj=(br-gt)[:,r].mean(1)-(br-gt)[:,l].mean(1);cj=(cr-gt)[:,r].mean(1)-(cr-gt)[:,l].mean(1)
                    row['original_seam_pair_partial'].append({**{k:v for k,v in pair.items() if k not in ('minus_mask','plus_mask')},
                        'baseline_signed_residual_jump_RGB':bj.tolist(),'candidate_signed_residual_jump_RGB':cj.tolist(),
                        'baseline_score':float(abs(bj).mean()),'candidate_score':float(abs(cj).mean()),'score_delta':float(abs(cj).mean()-abs(bj).mean())})
        for name,m in fixed.items():
            idx={'C2_model_positive':0,'semantic5_model_positive':4,'KEEP37_model_positive':1}[name]
            row['foreground'][name]={'pixels':int(m.sum()),'RGB_mean_abs_difference':stat(abs(cr-br).mean(0),m),
                'RGB_pixel_max_abs_difference':stat(abs(cr-br).max(0),m),'alpha_pixel_abs_difference':stat(abs(c['alpha']-b['alpha']),m),
                'depth_m_pixel_abs_difference':stat(abs(c['depth_m']-b['depth_m']),m),
                'own_group_alphaT_abs_difference':stat(abs(cg[idx]-bg[idx]),m),
                'own_group_alphaT_drop':stat(np.maximum(bg[idx]-cg[idx],0),m)}
            row['foreground'][name]['candidate_newroad_mass_sum']=float(cg[7][m].astype(np.float64).sum())
            if obs is not None:
                gt=obs['rgb'].transpose(2,0,1)
                row['foreground'][name].update(baseline_observed_RGB_MAE=stat(abs(br-gt).mean(0),m),candidate_observed_RGB_MAE=stat(abs(cr-gt).mean(0),m))
        if a.initial_case:
            ini=load(a.initial_case/f'frames/{raw:03d}/{state}_float.npz');fields=['alpha','depth_sum','depth_m','group_alpha_T','added_owner_rgb_T'];checks={}
            for field in fields:
                if field not in ini or field not in c:checks[field]={'status':'UNVERIFIED_MISSING_FIELD'};continue
                same=ini[field].shape==c[field].shape and ini[field].dtype==c[field].dtype
                checks[field]={'dtype_shape_equal':same,'bitwise_equal':bool(same and np.array_equal(ini[field],c[field])),
                    'max_abs_difference':float(abs(ini[field]-c[field]).max()) if same else None}
            row['global_geometry_vs_initial']=checks
            dy=(cr-ini['rgb']).transpose(1,2,0).astype(np.float64)@LUMA
            row['darkening_vs_C0']=[dark_components(dy,t,masks) for t in [-.01,-.05]] if raw in [94,105,128,137] else 'Only registered four-frame component display; full49 pixel statistics handled by fixed evaluator.'
            row['RGB_vs_C0_mean_abs']=float(abs(cr-ini['rgb']).astype(np.float64).mean())
        dy=(cr-br).transpose(1,2,0).astype(np.float64)@LUMA
        row['darkening_vs_latest']=[dark_components(dy,t,masks) for t in [-.01,-.05]] if raw in [94,105,128,137] else None
        if raw in SILVER:
            x0,y0,x1,y1=SILVER[raw];sl=(slice(y0,y1),slice(x0,x1));box=SILVER[raw]
            row['silver_ROI']={'bbox_xyxy':box,'pixels':(x1-x0)*(y1-y0),'baseline':component_stats(bg[:,y0:y1,x0:x1]),'candidate':component_stats(cg[:,y0:y1,x0:x1])}
            w,h=(x1-x0)*5,(y1-y0)*5;board=Image.new('RGB',(max(4*w,1200),h+95),'white');d=ImageDraw.Draw(board)
            d.text((8,5),f'{a.case_label} raw{raw} {state} silver ROI {box}; nearest x5; component alphaT is not RGB truth',fill='black')
            panels=[('Latest',np.rint(np.clip(br.transpose(1,2,0)[sl],0,1)*255).astype(np.uint8)),
                (a.case_label,np.rint(np.clip(cr.transpose(1,2,0)[sl],0,1)*255).astype(np.uint8)),
                ('Latest dominant component',COLORS[bg[:,y0:y1,x0:x1].argmax(0)]),('Candidate dominant component',COLORS[cg[:,y0:y1,x0:x1].argmax(0)])]
            for i,(label,im) in enumerate(panels):
                d.text((i*w+4,28),label,fill='black');board.paste(Image.fromarray(im).resize((w,h),Image.Resampling.NEAREST if hasattr(Image,'Resampling') else Image.NEAREST),(i*w,52))
            for i,name in enumerate(GROUPS):
                x=5+(i%4)*285;y=h+60+(i//4)*17;d.rectangle((x,y,x+10,y+10),fill=tuple(int(v) for v in COLORS[i]));d.text((x+14,y),name,fill='black')
            fp=a.output/f'raw{raw:03d}_{state}_silver_components.png';board.save(fp);report['figures'].append({'path':str(fp),'sha256':sha(fp),'size':board.size})
        report['records'].append(row)
    after=[snap(path) for path in sorted(paths)];assert before==after
    report['source_integrity']={'status':'PASS_ALL_INPUT_SIZE_MTIME_SHA_UNCHANGED','files':before}
    out=a.output.parent/(a.output.name+'.json');out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(out),'sha256':sha(out),'records':len(report['records'])}))


if __name__=='__main__':main()
