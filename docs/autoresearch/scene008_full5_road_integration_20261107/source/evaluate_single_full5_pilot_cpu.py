"""Saved-array CPU acceptance for the sole-Full5 staged scene (no renderer).

Example:
  python evaluate_single_full5_pilot_cpu.py --full5-case ROOT/baseline4 \
    --legacy-case OLDROOT/baseline4 --obs-root OBS --world-mask-root WORLD \
    --contract single_full5_staged20k_cpu_acceptance_draft_v1.json --output review.json

Optionally add --road-case ROOT/C4 and --evidence-map map.json.  The latter is
explicit receipt assertions, not guessed field names: {"assertions":[{"case":
"full5", "name":"parameters_unchanged", "path":"receipt.json", "pointer":
"/parameters_unchanged", "equals":true}, ...]}. Relative evidence paths resolve
against that case. Missing source/parity/identity evidence stays UNVERIFIED.
Group IDs must be saved in the NPZ or a case group_actor_ids.json; no index guess.
Legacy's frozen7-group ABI alone is explicit in the contract, not a Full5 ABI.
This script never releases training/49 or marks human/visual quality accepted.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from registered_seam_pairs import seam_pairs

RAWS=(94,105,128,137)
STATES=('keep','delete')
ACTORS={5,6,18,26,37}
LEGACY_GROUP_IDS=(6,37,-2,-1,5,18,26)
REQUIRED_EVIDENCE=('source_unchanged','parameters_unchanged','R_fixed','E37_only','native_rgb_parity')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def snap(p):
    p=Path(p);s=p.stat();return {'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':sha(p)}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'),parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def chw(a):
    a=np.asarray(a,dtype=np.float32)
    if a.shape==(320,480,3):a=a.transpose(2,0,1)
    assert a.shape==(3,320,480) and np.isfinite(a).all(),a.shape
    return a


def hw(a):
    a=np.asarray(a).squeeze();assert a.shape==(320,480) and np.isfinite(a).all();return a


def statistic(values,mask):
    a=np.asarray(values)[mask].astype(np.float64);n=int(mask.sum())
    return {'pixels':n,'sum':float(a.sum()),'mean':float(a.mean()) if n else None,'max':float(a.max()) if n else None}


def upper(mask):
    y,x=np.nonzero(mask)
    return mask&(np.arange(320)[:,None]<=y.min()+.25*(y.max()-y.min())) if len(y) else np.zeros_like(mask)


def boundary(mask):
    p=np.pad(mask,1,constant_values=False)
    eroded=p[1:-1,1:-1]&p[:-2,1:-1]&p[2:,1:-1]&p[1:-1,:-2]&p[1:-1,2:]
    return mask&~eroded


def camera_header(case,raw,data,inputs):
    row={}
    path=case/'frame_headers.json'
    if path.is_file():
        inputs.add(path);obj=read(path);rows=obj.get('frames',obj.get('headers',obj))
        if isinstance(rows,list):row=next((v for v in rows if int(v['raw'])==raw),{})
        elif isinstance(rows,dict):row=rows.get(str(raw),rows.get('%03d'%raw,{}))
    guard=case/f'frames/{raw:03d}/input_guard.json'
    if guard.is_file():
        inputs.add(guard);obj=read(guard);row={**obj,**row}
    row={**row,**{k:v for k,v in data.items() if k in ('raw','K','model_to_camera','W_model_to_camera','model_time','time','timestamp')}}
    if isinstance(row.get('camera'),dict):row={**row['camera'],**row}
    K=row.get('K');W=row.get('model_to_camera',row.get('W_model_to_camera'))
    t=row.get('model_time',row.get('time',row.get('timestamp')))
    if isinstance(K,dict):
        if all(k in K for k in ('fx','fy','cx','cy')):K=[[K['fx'],0,K['cx']],[0,K['fy'],K['cy']],[0,0,1]]
        else:K=None
    if K is None or W is None or t is None:return None
    K,W,t=np.asarray(K,np.float64),np.asarray(W,np.float64),float(np.asarray(t))
    assert K.shape==(3,3) and W.shape==(4,4) and np.isfinite(K).all() and np.isfinite(W).all() and np.isfinite(t)
    return K,W,t


def groups(case,data,label,inputs):
    array=data.get('group_alpha_T')
    ids=data.get('group_actor_ids',data.get('actor_channel_ids',data.get('identity_channel_ids')))
    if ids is None and (case/'group_actor_ids.json').is_file():
        path=case/'group_actor_ids.json';inputs.add(path);obj=read(path);ids=obj.get('group_actor_ids',obj) if isinstance(obj,dict) else obj
    if ids is None and label=='legacy':ids=LEGACY_GROUP_IDS
    if array is None or ids is None:return None,None
    ids=tuple(int(v) for v in np.asarray(ids).tolist());array=np.asarray(array,np.float32)
    assert array.shape==(len(ids),320,480) and np.isfinite(array).all()
    assert len(ids)==len(set(ids)) and ACTORS<=set(ids),(label,ids)
    return array,ids


def channel_metrics(array,ids,mask):
    if array is None:return {'status':'UNVERIFIED_NO_DECLARED_NATIVE_GROUPS'}
    n=int(mask.sum())
    if n==0:return {'pixels':0,'alphaT_mean':None,'argmax_rate':None}
    selected=array[:,mask];winner=selected.argmax(0)
    return {'pixels':n,'alphaT_mean':{str(a):float(selected[i].astype(np.float64).mean()) for i,a in enumerate(ids)},
            'argmax_rate':{str(a):float((winner==i).mean()) for i,a in enumerate(ids)},
            'meaning':'joint native alpha_T source-owner readout; not a semantic/body quality verdict'}


def signed_seams(rgb,gt,pairs):
    residual=rgb-gt;rows=[]
    for p in pairs:
        jump=residual[:,p['plus_mask']].mean(1)-residual[:,p['minus_mask']].mean(1)
        rows.append({k:v for k,v in p.items() if k not in ('minus_mask','plus_mask')})
        rows[-1].update(signed_residual_jump_rgb=jump.tolist(),score=float(np.abs(jump).mean()))
    return rows


def pointer(obj,path):
    for key in path.strip('/').split('/') if path.strip('/') else ():
        key=key.replace('~1','/').replace('~0','~');obj=obj[int(key)] if isinstance(obj,list) else obj[key]
    return obj


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--full5-case',type=Path,required=True);ap.add_argument('--legacy-case',type=Path,required=True)
    ap.add_argument('--road-case',type=Path);ap.add_argument('--obs-root',type=Path,required=True)
    ap.add_argument('--stage',choices=['structure','road_support','material_final'],default='structure')
    ap.add_argument('--world-mask-root',type=Path);ap.add_argument('--contract',type=Path,required=True)
    ap.add_argument('--evidence-map',type=Path);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();assert not args.output.exists(),'Keep previous independent evidence immutable'
    contract=read(args.contract);assert contract['pilot_raws']==list(RAWS)
    assert contract['schema']=='SINGLE_FULL5_STAGED20K_CPU_ACCEPTANCE_V2'
    assert not args.road_case or args.stage in ('road_support','material_final')
    cases={'legacy':args.legacy_case,'full5':args.full5_case}
    if args.road_case:cases['full5_C']=args.road_case
    inputs={args.contract,Path(__file__),Path(__file__).with_name('registered_seam_pairs.py')}
    assertions=[]
    if args.evidence_map:inputs.add(args.evidence_map);assertions=read(args.evidence_map)['assertions']
    for raw in RAWS:
        inputs.add(args.obs_root/f'{raw:03d}.npz')
        for case in cases.values():
            inputs.add(case/f'frames/{raw:03d}/keep_float.npz');inputs.add(case/f'frames/{raw:03d}/delete_float.npz')
    if args.world_mask_root:
        inputs.update([args.world_mask_root/'registry.json',args.world_mask_root/'primary_raw128/fixed_masks_raw128.npz'])
        inputs.update(args.world_mask_root/f'{raw:03d}.npz' for raw in RAWS)
    for a in assertions:
        p=Path(a['path']);inputs.add(p if p.is_absolute() else cases[a['case']]/p)
    # Known companion inputs are fingerprinted before the readout, even if absent
    # optional evidence would leave acceptance UNVERIFIED.
    for case in cases.values():
        for p in [case/'frame_headers.json',case/'group_actor_ids.json']+[case/f'frames/{r:03d}/input_guard.json' for r in RAWS]:
            if p.is_file():inputs.add(p)
    before={str(p):snap(p) for p in inputs}
    checks=[];records=[];cached={};domains={};seam_cache={}
    def check(name,value,threshold,**ctx):
        checks.append(dict(name=name,value=value,threshold=threshold,passed=None if value is None else bool(value<=threshold),**ctx))
    for raw in RAWS:
        obs=load(args.obs_root/f'{raw:03d}.npz');gt=chw(obs['rgb']);a5=obs['actor_labels']==1;a6=obs['visible'][0];a37=obs['visible'][1]
        base_keep=load(args.legacy_case/f'frames/{raw:03d}/keep_float.npz')
        legacy_g,legacy_ids=groups(args.legacy_case,base_keep,'legacy',inputs)
        if legacy_g is None:raise ValueError('The frozen legacy37 footprint needs its declared native group1 readout')
        footprint=(obs['actor_labels']==5)|(legacy_g[legacy_ids.index(37)]>.5)
        masks={'a5_RGB':a5,'a5_upper_RGB':upper(a5),'a5_boundary_RGB':boundary(a5),
               'a6_RGB':a6,'a6_full_body_RGB':obs['actor_labels']==4,'a37_RGB':a37,
               'a37_full_body_RGB':obs['actor_labels']==5,'DELETE_footprint_RGB':footprint,
               'protected_nonroad_RGB':obs['protect']&~a6&~a37&~obs['road']&~obs['road_edit'],
               'known_environment_RGB':obs['protect']&(obs['actor_labels']==0)&~obs['road']&~obs['road_edit'],
               'original_road_RGB':obs['road']}
        guard=~obs['road']&~obs['road_edit']&(obs['protect']|a5|a6|a37)
        domains[raw]=(masks,guard)
        pairs=seam_pairs(raw,args.world_mask_root) if args.world_mask_root else []
        for label,case in cases.items():
            for state in STATES:
                data=load(case/f'frames/{raw:03d}/{state}_float.npz');rgb=chw(data['rgb']);alpha=hw(data['alpha'])
                depth=hw(data['depth_m']) if 'depth_m' in data else None
                g,ids=groups(case,data,label,inputs)
                row={'raw':raw,'state':state,'case':label,'metrics':{},'group_actor_ids':list(ids) if ids else None,
                     'native_rgb_min':float(rgb.min()),'native_rgb_max':float(rgb.max()),
                     'alpha':{name:statistic(alpha,m) for name,m in masks.items()}}
                error=np.abs(rgb-gt).mean(0)
                row['metrics']={name:statistic(error,m) for name,m in masks.items()}
                dm=obs['depth_valid'][0].astype(bool)
                row['metrics']['a6_depth_m']=statistic(np.abs(depth-obs['depth_m'][0]),dm) if depth is not None else {'pixels':int(dm.sum()),'mean':None,'sum':None,'max':None}
                row['identity']={name:channel_metrics(g,ids,m) for name,m in masks.items() if name in ('a5_RGB','a5_upper_RGB','a6_RGB','DELETE_footprint_RGB')}
                header=camera_header(case,raw,data,inputs)
                row['camera_header_status']='UNVERIFIED_MISSING_NATIVE_HEADER' if header is None else 'CHECKED'
                if label!='legacy':
                    check('camera_K_max_abs',float(abs(header[0]-obs['K']).max()) if header else None,1e-6,case=label,raw=raw,state=state)
                    check('camera_W_max_abs',float(abs(header[1]-obs['model_to_camera']).max()) if header else None,1e-5,case=label,raw=raw,state=state)
                    check('camera_time_abs',abs(header[2]-float(obs['model_time'])) if header else None,1e-8,case=label,raw=raw,state=state)
                    check('identity_partition_alpha_max_abs',float(abs(g.sum(0)-alpha).max()) if g is not None else None,1e-5,case=label,raw=raw,state=state)
                    if state=='delete' and label=='full5':
                        check('native_SOURCE37_filter_alpha_T_max_MECHANICAL_ONLY',float(g[ids.index(37)].max()) if g is not None else None,1e-8,case=label,raw=raw,state=state)
                    elif state=='delete':
                        selected=data.get('query_selected_alpha_T')
                        check('actual_QUERY_selector_alpha_T_max',float(np.asarray(selected).max()) if selected is not None else None,1e-8,case=label,raw=raw,state=state)
                row['seams']=signed_seams(rgb,gt,pairs)
                records.append(row);cached[(label,raw,state)]=(row,rgb,alpha,depth,g,ids)
                seam_cache[(label,raw,state)]=row['seams']
    comparisons=[];reference_flags=[];road_checks=[]
    for raw in RAWS:
        masks,guard=domains[raw]
        for state in STATES:
            for candidate,base,role in [('full5','legacy','REFERENCE_DIFFERENCE_NOT_EQUIVALENCE_PASS')]+([('full5_C','full5','ROAD_INCREMENT_ON_THE_SOLE_SCENE')] if args.road_case else []):
                cr,ci,ca,cd,cg,cids=cached[candidate,raw,state];br,bi,ba,bd,bg,bids=cached[base,raw,state]
                diffs={}
                for name,after in cr['metrics'].items():
                    before_value=br['metrics'][name]['mean'];after_value=after['mean']
                    diffs[name]={'pixels':after['pixels'],'baseline':before_value,'candidate':after_value,'delta':after_value-before_value if before_value is not None and after_value is not None else None}
                comparisons.append({'raw':raw,'state':state,'candidate':candidate,'baseline':base,'role':role,'metrics':diffs,
                    'RGB_pixel_max_abs_difference':float(abs(ci-bi).max()),
                    'alpha_pixel_max_abs_difference':float(abs(ca-ba).max()),
                    'a6_measured_depth_readout':'joint conditional depth, not an unsaved actor-specific depth component',
                    'fixed_DELETE_footprint_RGB_mean_abs_movement':statistic(abs(ci-bi).mean(0),masks['DELETE_footprint_RGB']) if state=='delete' else None})
                target=reference_flags if candidate=='full5' else road_checks
                names=('a5_RGB','a5_upper_RGB','a6_RGB','a6_depth_m','protected_nonroad_RGB','original_road_RGB') if candidate=='full5' else ('a5_RGB','a5_upper_RGB','a6_RGB','a6_depth_m','known_environment_RGB')
                for name in names+(('a37_RGB',) if state=='keep' else ()):
                    value=diffs[name]['delta'];threshold=1e-5 if name=='a6_depth_m' else 1e-6
                    target.append(dict(name=name,raw=raw,state=state,value=value,threshold=threshold,passed=None if value is None else bool(value<=threshold),role=role))
                if candidate=='full5_C':
                    # The broad static/road guard is reported, not a blanket veto
                    # against intended background/silver reconstruction in v2.
                    for name,value,threshold in [('nonroad_guard_alpha_pixel_max_abs',statistic(abs(ca-ba),guard)['max'],1e-6),
                        ('nonroad_guard_depth_pixel_max_abs_m',statistic(abs(cd-bd),guard)['max'] if cd is not None and bd is not None else None,1e-5)]:
                        comparisons[-1].setdefault('broad_guard_report_only',[]).append(dict(name=name,raw=raw,state=state,value=value,historical_threshold=threshold,role='REPORT_ONLY_NOT_STRUCTURE_VETO'))
                    retained=np.zeros((320,480),bool)
                    obs=load(args.obs_root/f'{raw:03d}.npz')
                    for actor,lab in [(5,1),(26,2),(18,3),(6,4)]+([(37,5)] if state=='keep' else []):
                        m=obs['actor_labels']==lab;retained|=m
                        if actor==6:m=m|obs['visible'][0];retained|=m
                        if actor==37:m=m|obs['visible'][1];retained|=m
                        value=statistic(np.maximum(bg[bids.index(actor)]-cg[cids.index(actor)],0),m)['max'] if bg is not None and cg is not None else None
                        road_checks.append(dict(name='retained_actor%d_alphaT_max_drop'%actor,raw=raw,state=state,value=value,threshold=1e-6,passed=None if value is None else value<=1e-6))
                    value=statistic(abs(cd-bd),retained)['max'] if cd is not None and bd is not None else None
                    road_checks.append(dict(name='retained_real_foreground_depth_pixel_max_abs_m',raw=raw,state=state,value=value,threshold=1e-5,passed=None if value is None else value<=1e-5))
    pooled=[]
    for label in cases:
        for state in STATES:
            for kind,edge in [('world','near_x'),('world','lateral_y'),('primary128','near_x')]:
                rows=[p for raw in RAWS for p in seam_cache[label,raw,state] if p['kind']==kind and p['edge']==edge]
                pooled.append(dict(case=label,state=state,kind=kind,edge=edge,bins=len(rows),mean=float(np.mean([p['score'] for p in rows])) if rows else None))
    if args.road_case and args.stage=='material_final':
        pm={(r['case'],r['state'],r['kind'],r['edge']):r for r in pooled}
        for state in STATES:
            for kind,edge in [('world','near_x'),('world','lateral_y'),('primary128','near_x')]:
                b=pm['full5',state,kind,edge]['mean'];c=pm['full5_C',state,kind,edge]['mean'];delta=c-b if b is not None and c is not None else None
                road_checks.append(dict(name=kind+':'+edge+':pooled_delta',state=state,value=delta,threshold=1e-6,passed=None if delta is None else delta<=1e-6))
                if kind=='primary128':
                    for name,value,threshold in [('primary_absolute_reduction',delta,-1e-4),('primary_relative_reduction',delta/b if delta is not None and b>0 else None,-.05)]:
                        road_checks.append(dict(name=name,state=state,value=value,threshold=threshold,passed=None if value is None else value<=threshold))
    # Same-scene KEEP->DELETE on observed retained actor pixels. This directly
    # exposes source37 support that actually reconstructs another visible car.
    retained_query_diagnostics=[]
    for label in cases:
        if label=='legacy':continue
        for raw in RAWS:
            obs=load(args.obs_root/f'{raw:03d}.npz')
            kr,ki,ka,kd,kg,kids=cached[label,raw,'keep'];dr,di,da,dd,dg,dids=cached[label,raw,'delete']
            for actor,lab in [(5,1),(26,2),(18,3),(6,4)]:
                m=obs['actor_labels']==lab;n=int(m.sum())
                if actor==6:m=m|obs['visible'][0];n=int(m.sum())
                gt=chw(obs['rgb']);ke=statistic(abs(ki-gt).mean(0),m);de=statistic(abs(di-gt).mean(0),m)
                delta=de['mean']-ke['mean'] if n else None
                retained_query_diagnostics.append(dict(case=label,raw=raw,actor=actor,pixels=n,KEEP_observed_RGB_MAE=ke['mean'],DELETE_observed_RGB_MAE=de['mean'],
                    delta_observed_RGB_MAE=delta,RGB_KD_mean_abs=statistic(abs(di-ki).mean(0),m)['mean'],
                    KEEP_source37_alphaT_mean=statistic(kg[kids.index(37)],m)['mean'] if kg is not None else None,
                    KEEP_source37_argmax_rate=float((kg[:,m].argmax(0)==kids.index(37)).mean()) if kg is not None and n else None,
                    no_regression_reference_flag=None if delta is None else delta<=1e-6,
                    role='SAME_SCENE_QUERY_SEMANTIC_DIAGNOSTIC; ancestry integrity does not imply semantic correctness'))
    evidence=[]
    for a in assertions:
        p=Path(a['path']);p=p if p.is_absolute() else cases[a['case']]/p
        try:actual=pointer(read(p),a['pointer']);passed=bool(actual==a['equals'])
        except (KeyError,IndexError,ValueError,TypeError):actual=None;passed=False
        evidence.append(dict(case=a['case'],name=a['name'],path=str(p),sha256=sha(p),pointer=a['pointer'],actual=actual,expected=a['equals'],passed=passed))
    missing={label:[name for name in REQUIRED_EVIDENCE if not any(e['case']==label and e['name']==name and e['passed'] for e in evidence)] for label in cases if label!='legacy'}
    after={str(p):snap(p) for p in inputs};assert before==after,'Inputs changed while CPU audit was reading'
    failed=[c for c in checks+road_checks if c['passed'] is False];unsupported=[c for c in checks+road_checks if c['passed'] is None]
    result={'schema':'SINGLE_FULL5_SAVED_PILOT_CPU_V1','status':'ARRAY_REVIEW_FAILED_GATES' if failed else 'ARRAY_READOUT_COMPLETE_AWAITING_METADATA_OR_ROOT_STRUCTURE_REVIEW',
        'generated_utc':datetime.now(timezone.utc).isoformat(),'contract_sha256':sha(args.contract),'contract_status':contract['status'],
        'scope':'Sole existing Full5@10K migration and road integration; not a road-integrated0-to10K training claim. Original LR horizon30K remains historical.',
        'stage':args.stage,'raws':list(RAWS),'states':list(STATES),'records':records,'comparisons':comparisons,'pooled_seams':pooled,
        'engineering_checks':checks,'road_increment_checks':road_checks,'legacy_reference_no_regression_flags':reference_flags,
        'same_scene_retained_actor_query_diagnostics':retained_query_diagnostics,
        'material_final_seam_targets_not_stage0_hard_gates':args.stage!='material_final',
        'failed_checks':failed,'unsupported_checks':unsupported,'explicit_receipt_assertions':evidence,'missing_required_metadata_evidence':missing,
        'source_integrity':{'status':'PASS_ALL_READ_INPUT_SIZE_MTIME_SHA_UNCHANGED','files':list(before.values())},
        'new_CPU_audit_gpu_forwards':0,'new_CPU_audit_backwards':0,'new_CPU_audit_optimizer_updates':0,
        'full5_structure_accepted':False,'material_stage_released':False,'all49_released':False,'quality_accepted':False,'human_verdict':None,
        'limits':['RGB metrics retain native floats; display clipping is not applied to scores.',
            'Legacy a6/C2 and native Full5 owner6 differ; reference deltas cannot be labelled equivalence or lossless.',
            'Observed includes actor37; full37/delete-footprint comparison to observed is diagnostic, not clean-delete quality.',
            'Owner alpha and partition checks are identity mechanics, not a normal-roof or semantic-quality pass.',
            'Root must actually inspect the four frames and resolve visible a6/structure regressions before any material release.']}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(args.output),'status':result['status'],'failed':len(failed),'unsupported':len(unsupported),'missing_metadata':missing,'material_stage_released':False}))


if __name__=='__main__':main()
