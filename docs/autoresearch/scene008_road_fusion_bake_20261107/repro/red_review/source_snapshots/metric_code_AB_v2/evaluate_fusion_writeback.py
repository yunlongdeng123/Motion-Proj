"""Three-layer, fixed-domain CPU evaluation of persistent colour writeback.

1 target vs original visible measurements; 2 rendered writeback vs desired
full-composite target; 3 render vs original road/vehicle/protection domains.
Uses immutable masks and all registered per-edge bins, never outcome-selected.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,time
from collections import defaultdict
from pathlib import Path
import numpy as np
from prepare_world_masks49 import pair_summary

EDGES={'near_x':1,'far_x':1,'lateral_y':0}
GEOMETRY=('alpha','depth_m','group_alpha_T','newroad_alpha_T','oldroad_alpha_T')
HELD9=(89,95,101,105,113,121,125,130,137)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(p):
    with np.load(p,allow_pickle=False) as a:return {k:a[k] for k in a.files}
def chw(a):
    a=np.asarray(a)
    if a.shape==(320,480,3):a=a.transpose(2,0,1)
    if a.shape!=(3,320,480):raise ValueError(('Expected native CHW/HWC RGB',a.shape))
    return a
def stat(a):
    a=np.asarray(a,float);a=a[np.isfinite(a)]
    return dict(count=len(a),sum=float(a.sum()),mean=float(a.mean()) if len(a) else None,min=float(a.min()) if len(a) else None,max=float(a.max()) if len(a) else None)
def error_stats(a,b,mask,valid=None):
    fixed=int(mask.sum());v=mask if valid is None else mask&valid
    out=stat(np.abs(a[:,v]-b[:,v]).mean(0))
    out.update(fixed_domain_pixels=fixed,scored_pixels=int(v.sum()),unscored_pixels=fixed-int(v.sum()))
    return out


def edge_metrics(image,reference,masks,registration,valid=None,replace=None):
    residual=image-reference;result={};xyz=masks['local_xyz']
    for edge,ta in EDGES.items():
        mm=masks[edge+'_minus'];mp=masks[edge+'_plus'];selected=mm|mp
        bins=np.zeros(mm.shape,np.int32);bins[selected]=np.floor(xyz[...,ta][selected]/.2).astype(np.int32)
        pairs=[];incomplete=[]
        for r in registration[edge]['bins']:
            b=r['tangent_bin'];minus=mm&(bins==b);plus=mp&(bins==b)
            assert int(minus.sum())==r['minus_count'] and int(plus.sum())==r['plus_count']
            if valid is not None and (not valid[minus].all() or not valid[plus].all()):
                incomplete.append(dict(tangent_bin=b,valid_minus=int((minus&valid).sum()),valid_plus=int((plus&valid).sum()),**{k:r[k] for k in ('minus_count','plus_count')}));continue
            jump=residual[:,plus].mean(1)-residual[:,minus].mean(1)
            entry=dict(**r,signed_residual_jump_rgb=jump.astype(float).tolist(),abs_channel_jump=float(np.abs(jump).mean()))
            if replace is not None:
                nr,nl=int((minus&replace).sum()),int((plus&replace).sum())
                entry.update(minus_replace_count=nr,plus_replace_count=nl,minus_identity_count=int(minus.sum())-nr,plus_identity_count=int(plus.sum())-nl,
                    target_composition='IDENTITY_ONLY' if nr+nl==0 else 'ATLAS_ONLY' if nr+nl==int(minus.sum()+plus.sum()) else 'MIXED_ATLAS_AND_ORIGINAL_RGB')
            pairs.append(entry)
        complete=len(pairs)==len(registration[edge]['bins'])
        result[edge]=dict(registered_paired_bins=len(registration[edge]['bins']),scored_paired_bins=len(pairs),
            status='UNSUPPORTED' if not registration[edge]['bins'] else 'COMPLETE' if complete else 'INCOMPLETE_TARGET_COVERAGE',
            equal_bin_abs_residual_jump_mean=float(np.mean([p['abs_channel_jump'] for p in pairs])) if pairs and complete else None,
            conditional_complete_bin_mean=float(np.mean([p['abs_channel_jump'] for p in pairs])) if pairs else None,
            bins=pairs,incomplete_bins=incomplete)
        if replace is not None:
            result[edge].update(band_minus_replace=int((mm&replace).sum()),band_plus_replace=int((mp&replace).sum()),
                matched_minus_replace=sum(p['minus_replace_count'] for p in pairs),matched_plus_replace=sum(p['plus_replace_count'] for p in pairs),
                identity_only_bins=sum(p['target_composition']=='IDENTITY_ONLY' for p in pairs),mixed_target_bins=sum(p['target_composition']=='MIXED_ATLAS_AND_ORIGINAL_RGB' for p in pairs),
                atlas_only_bins=sum(p['target_composition']=='ATLAS_ONLY' for p in pairs))
    return result


def gradients(image,reference,mask,valid):
    records=[]
    for axis in (0,1):
        m0=mask[:-1] if axis==0 else mask[:,:-1];m1=mask[1:] if axis==0 else mask[:,1:]
        v0=valid[:-1] if axis==0 else valid[:,:-1];v1=valid[1:] if axis==0 else valid[:,1:]
        fixed=m0&m1;scored=fixed&v0&v1
        da=np.diff(image,axis=axis+1);db=np.diff(reference,axis=axis+1)
        s=stat(np.abs(da[:,scored]-db[:,scored]).mean(0));s.update(axis='vertical' if axis==0 else 'horizontal',fixed_pairs=int(fixed.sum()),scored_pairs=int(scored.sum()));records.append(s)
    return records


def equality(a,b):
    if a is None or b is None:return dict(status='NOT_SAVED_IN_FLOAT_NPZ')
    exact=a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)
    return dict(status='BITWISE_EQUAL' if exact else 'DIFFERENT',dtype_before=str(a.dtype),dtype_after=str(b.dtype),shape_before=list(a.shape),shape_after=list(b.shape),
        max_abs_diff=float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.shape==b.shape else None)


def ray_registry(previous):
    a=read(previous/'red/raw128_contact_audit_v2.json')
    rows={128:[dict(pixel_xy=r['pixel_xy'],ordering={1:'chart_first',2:'ground_first',3:'near_contact_005m',0:'uncertain'}[r['ordering_matte_001']],V44_assetW=r['baseline_assetW']) for r in a['evidence_pixels'] if r['baseline_assetW']>.9]}
    old=previous.parent/'scene008_road_r0_v44_20261107_r1/review/asset_ground_conflict/canonical_chart_35ray_first_hit_probe.json'
    b=read(old);rows[130]=[dict(pixel_xy=r['pixel_xy'],ordering=r['matte_thresholds']['0.01']['ordering'],V44_assetW=r['baseline_assetW']) for r in b['pixels']]
    assert len(rows[128])==36 and len(rows[130])==35
    return rows,{str(previous/'red/raw128_contact_audit_v2.json'):sha(previous/'red/raw128_contact_audit_v2.json'),str(old):sha(old)}


def red_rays(base,candidate,rays):
    if 'group_alpha_T' not in base or 'group_alpha_T' not in candidate:return dict(status='FULL_FIELDS_NOT_SAVED')
    pixels=[]
    for r in rays:
        x,y=r['pixel_xy'];b=float(base['group_alpha_T'][0,y,x]);c=float(candidate['group_alpha_T'][0,y,x])
        pixels.append(dict(**r,baseline_assetW=b,candidate_assetW=c,assetW_change=c-b,RGB_abs_change=float(np.abs(candidate['rgb'][:,y,x]-base['rgb'][:,y,x]).mean())))
    summary=[]
    for group in ('chart_first','ground_first','near_contact_005m','uncertain'):
        q=[r for r in pixels if r['ordering']==group]
        if not q:continue
        summary.append(dict(group=group,rays=len(q),assetW_change=stat([r['assetW_change'] for r in q]),
            newly_exposed_assetW_gt001=sum(r['baseline_assetW']<=.01 and r['candidate_assetW']>.01 for r in q),
            remaining_V44_loss_gt05=sum(r['V44_assetW']-r['candidate_assetW']>.5 for r in q)))
    return dict(state='delete',status='COLOUR_ONLY_EXPECT_ZERO_WEIGHT_CHANGE',summary=summary,pixels=pixels,
        scope='Same current C2/ground internal geometry reference; shares BUILD LiDAR with road. Unknown hidden truth; RGB change is not wheel correctness.')


def receipt_evidence(receipt,raw,state,method):
    if not receipt:return dict(status='NO_NATIVE_RECEIPT')
    rows=[r for r in receipt.get('records',[]) if r.get('raw')==raw and r.get('state')==state]
    base=next((r for r in rows if r.get('variant')=='baseline'),None);candidate=next((r for r in rows if r.get('variant')==method),None)
    if not base or not candidate:return dict(status='NO_MATCHING_NATIVE_RECORD')
    equal={k:base.get('geometry_readout_hashes',{}).get(k)==candidate.get('geometry_readout_hashes',{}).get(k) and k in base.get('geometry_readout_hashes',{}) for k in GEOMETRY}
    return dict(status='RECEIPT_HASH_COMPARISON',fields_equal=equal,vehicle_row_hash_equal=base.get('asset_rows_hash')==candidate.get('asset_rows_hash'),
        frozen_asset_premultiplied_RGB_exact=candidate.get('frozen_asset_premultiplied_RGB_exact'),frozen_a37_premultiplied_RGB_exact=candidate.get('frozen_a37_premultiplied_RGB_exact'),
        baseline_parent_PNG_exact=base.get('baseline_parent_PNG_exact'))


def replace_alpha_context(base,mask):
    if 'newroad_alpha_T' not in base:return dict(status='FULL_FLOAT_READOUT_NOT_SAVED')
    W=base['newroad_alpha_T'];zero=mask&(W==0);q=dict(status='BASELINE_COMPOSITING_MASS_NOT_FIRST_HIT',pixels=int(mask.sum()),Wnew_zero=int(zero.sum()),Wnew_lt001=int((mask&(W<.01)).sum()),Wnew=stat(W[mask]))
    if 'group_alpha_T' in base:
        group=base['group_alpha_T'];alpha=base['alpha'][0] if base['alpha'].ndim==3 else base['alpha']
        components=np.stack([group[0],group[1],base['oldroad_alpha_T'],group[3],1-alpha])
        owner=np.argmax(components,axis=0);names=('C2_asset','a37','oldroad','native_static','residual_environment')
        q['Wnew_zero_maximum_mass_component_counts']={name:int((zero&(owner==i)).sum()) for i,name in enumerate(names)}
    return q


def receipt_pool_check(receipt):
    if not receipt or 'records' not in receipt:return dict(status='NO_MATCHING_NATIVE_RECEIPT_SCHEMA')
    totals=defaultdict(lambda:[0.,0]);mismatches=[]
    for r in receipt['records']:
        for metric,v in r['metrics'].items():
            if not v:continue
            t=totals[r['state'],r['variant'],metric];t[0]+=v['sum'];t[1]+=v['count']
    for r in receipt['pooled']:
        s,n=totals[r['state'],r['variant'],r['metric']]
        if n!=r['count'] or abs(s-r['sum'])>1e-6 or (n and abs(s/n-r['mean'])>1e-12):mismatches.append(dict(state=r['state'],variant=r['variant'],metric=r['metric']))
    return dict(status='RECOMPUTED_NATIVE_RECEIPT_SUM_COUNT',records=len(receipt['records']),pooled_rows=len(receipt['pooled']),mismatches=mismatches,
        evidence='Scalar receipt recomputation; not independent full-float depth reload for the44 RGB-only frames.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--previous-run',type=Path,required=True)
    p.add_argument('--mask-registry',type=Path,required=True);p.add_argument('--readouts',type=Path,required=True)
    p.add_argument('--baseline-pattern',default='baseline_{state}.npz');p.add_argument('--methods',default='direct,fusion');p.add_argument('--method-pattern',default='{method}_{state}.npz')
    p.add_argument('--target-root',type=Path,action='append',default=[]);p.add_argument('--target-manifest',type=Path);p.add_argument('--source-audit',type=Path);p.add_argument('--raws',default='');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();start=time.time()
    if a.output.exists():raise FileExistsError(a.output)
    reg=read(a.mask_registry);frame_meta={r['raw']:r for r in reg['frames']};maskroot=a.mask_registry.parent
    raws=[int(x) for x in a.raws.split(',')] if a.raws else reg['raws'];methods=a.methods.split(',')
    scope='ALL49' if raws==list(range(89,138)) else f'SELECTED_{len(raws)}VIEWS'
    receipt=read(a.readouts/'receipt.json') if (a.readouts/'receipt.json').exists() else None
    rays,ray_sources=ray_registry(a.previous_run);source_sha={str(a.mask_registry):sha(a.mask_registry),**ray_sources};rows=[];pools=defaultdict(lambda:[0.,0])
    target_manifest=a.target_manifest or a.run/'targets/target_manifest.json'
    target_contract=read(target_manifest) if a.target_root and target_manifest.exists() else None
    if target_contract:
        assert target_contract['fusion_ready'] is True
        source_sha[str(target_manifest)]=sha(target_manifest)
    audit_path=a.source_audit or a.run/'red_review/source_and_fit_contract_audit_v1.json'
    audit=read(audit_path) if audit_path.exists() else None
    audit_verified=bool(audit and audit['status']=='DATA_AND_CONTRACT_CHECKS_PASS_WITH_LIMITATIONS' and target_contract and audit['input_sha256'].get(str(target_manifest))==sha(target_manifest))
    if audit:source_sha[str(audit_path)]=sha(audit_path)
    # Verify every requested mask against the frozen registry before target or
    # native candidate reads. The primary bbox remains a separate population.
    for raw in raws:assert sha(maskroot/frame_meta[raw]['mask_file'])==frame_meta[raw]['mask_sha256']
    primary=load(maskroot/'primary_raw128/fixed_masks_raw128.npz')
    primary_world={'local_xyz':primary['local_xyz']}
    primary_world.update({edge+'_'+side:primary['band_'+edge+'_'+side] for edge in EDGES for side in ('minus','plus')})
    primary_pairs=pair_summary(primary_world)
    for raw in raws:
        meta=frame_meta[raw];masks=load(maskroot/meta['mask_file']);obs=load(Path(meta['obs_path']));observed=chw(obs['rgb'])
        domains={'original_road':obs['road'].astype(bool),'protected':obs['protect'].astype(bool)&~obs['visible'].any(0),'a6_visible':obs['visible'][0].astype(bool)}
        domains.update({k:v for k,v in masks.items() if v.dtype==bool})
        if raw==128:domains.update({'primary_bbox_'+k:v for k,v in primary.items() if v.dtype==bool})
        target_path=next((d/f'raw{raw:03d}.npz' for d in a.target_root if (d/f'raw{raw:03d}.npz').exists()),None)
        targets=load(target_path) if target_path else None
        if target_path:source_sha[str(target_path)]=sha(target_path)
        for state in ('keep','delete'):
            bp=a.readouts/f'{raw:03d}'/a.baseline_pattern.format(state=state);base=load(bp);base['rgb']=chw(base['rgb']);source_sha[str(bp)]=sha(bp)
            current_domains=dict(domains)
            if state=='keep':current_domains['a37_visible']=obs['visible'][1].astype(bool)
            for method in methods:
                cp=a.readouts/f'{raw:03d}'/a.method_pattern.format(method=method,state=state);candidate=base if cp==bp else load(cp);candidate['rgb']=chw(candidate['rgb']);source_sha[str(cp)]=sha(cp)
                row=dict(raw=raw,state=state,method=method,subset='HELD9_PREEXPOSED_SCENE' if raw in HELD9 else 'BUILD40',
                    layer3_original_observations={},world_boundaries={},target_present=targets is not None,
                    saved_float_invariants={k:equality(base.get(k),candidate.get(k)) for k in GEOMETRY},native_receipt=receipt_evidence(receipt,raw,state,method))
                for name,mask in current_domains.items():
                    b=error_stats(base['rgb'],observed,mask);c=error_stats(candidate['rgb'],observed,mask)
                    row['layer3_original_observations'][name]=dict(baseline=b,candidate=c,mean_delta=None if not c['count'] else c['mean']-b['mean'])
                    for variant,v in (('baseline',b),('candidate',c)):
                        for subset in (scope,row['subset']):
                            q=pools[state,method,subset,name,variant];q[0]+=v['sum'];q[1]+=v['count']
                row['world_boundaries']=dict(baseline=edge_metrics(base['rgb'],observed,masks,meta['edges']),candidate=edge_metrics(candidate['rgb'],observed,masks,meta['edges']))
                if raw==128:row['primary_bbox_boundaries']=dict(baseline=edge_metrics(base['rgb'],observed,primary_world,primary_pairs),candidate=edge_metrics(candidate['rgb'],observed,primary_world,primary_pairs))
                if targets is not None:
                    key=method+'_rgb'
                    if key not in targets:row['layer1_target']=dict(status='NO_METHOD_TARGET_KEY',requested=key)
                    else:
                        desired=chw(targets[key]);finite=np.isfinite(desired).all(0);fit=targets['fit_mask'].astype(bool)
                        replace=targets.get('fusion_replace_mask',targets.get('replace_mask',fit)).astype(bool)
                        d=dict(current_domains);d['common_legal_fit']=fit;d['fusion_replace']=replace
                        # Targets are encoded sRGB desired full-composite RGB,
                        # state-shared. Neither target is called hidden GT.
                        row['layer1_target']=dict(space='DESIRED_FULL_COMPOSITED_IMAGE_RGB',encoding='encoded_sRGB',state_shared=True,
                            source_contract_pending_audit=not audit_verified,source_contract_audit_sha256=sha(audit_path) if audit else None,
                            out_of_range_channel_values=int(((desired<0)|(desired>1)).sum()),nonfinite_pixels=int((~finite).sum()),
                            fidelity_to_original_visible={n:error_stats(desired,observed,m,finite) for n,m in d.items()},
                            replace_only_fidelity_to_original={n:error_stats(desired,observed,m&replace,finite) for n,m in d.items()},
                            domain_composition={n:dict(fixed_pixels=int(m.sum()),atlas_replace_pixels=int((m&replace).sum()),identity_original_RGB_pixels=int((m&~replace).sum()),fit_pixels=int((m&fit).sum())) for n,m in d.items()},
                            world_boundaries=edge_metrics(desired,observed,masks,meta['edges'],finite,replace),
                            measured_gradient_fidelity=gradients(desired,observed,domains['world_patch_clean_road'],finite),
                            provenance_mask_counts={k:int(targets[k].sum()) for k in ('fit_mask','fusion_replace_mask','atlas_support_mask','imputed_mask') if k in targets})
                        row['layer2_writeback_to_target']=dict(error={n:error_stats(candidate['rgb'],desired,m,finite) for n,m in d.items()},
                            replace_only_error={n:error_stats(candidate['rgb'],desired,m&replace,finite) for n,m in d.items()},
                            world_boundaries=edge_metrics(candidate['rgb'],desired,masks,meta['edges'],finite,replace),
                            exact_target_domain='Target is desired joint RGB, not alpha-premultiplied road colour or intrinsic albedo.')
                        if raw==128:
                            row['layer1_target']['primary_bbox_boundaries']=edge_metrics(desired,observed,primary_world,primary_pairs,finite,replace)
                            row['layer2_writeback_to_target']['primary_bbox_boundaries']=edge_metrics(candidate['rgb'],desired,primary_world,primary_pairs,finite,replace)
                        row['baseline_replace_compositing_context']=replace_alpha_context(base,replace)
                if raw in rays and state=='delete':row['fixed_red_rays']=red_rays(base,candidate,rays[raw])
                rows.append(row)
        print(json.dumps(dict(raw=raw,rows=len(rows),seconds=time.time()-start)),flush=True)
    pooled=[dict(state=k[0],method=k[1],subset=k[2],domain=k[3],variant=k[4],sum=s,count=n,mean=s/n if n else None) for k,(s,n) in pools.items()]
    seam_pool=defaultdict(list)
    for row in rows:
        sets={'baseline':row['world_boundaries']['baseline'],'candidate':row['world_boundaries']['candidate']}
        if 'world_boundaries' in row.get('layer1_target',{}):sets['target_vs_observation']=row['layer1_target']['world_boundaries']
        if 'world_boundaries' in row.get('layer2_writeback_to_target',{}):sets['candidate_vs_target']=row['layer2_writeback_to_target']['world_boundaries']
        for variant,edges in sets.items():
            for edge,value in edges.items():
                for subset in (scope,row['subset']):seam_pool[row['state'],row['method'],subset,variant,edge].append(value)
    seam_pooled=[]
    for key,values in seam_pool.items():
        complete=[v for v in values if v['status']=='COMPLETE'];bins=[b['abs_channel_jump'] for v in complete for b in v['bins']]
        seam_pooled.append(dict(state=key[0],method=key[1],subset=key[2],variant=key[3],edge=key[4],
            views=len(values),supported_complete_views=len(complete),unsupported_views=sum(v['status']=='UNSUPPORTED' for v in values),incomplete_views=sum(v['status']=='INCOMPLETE_TARGET_COVERAGE' for v in values),
            scored_paired_bins=len(bins),equal_bin_abs_residual_jump=stat(bins),
            equal_supported_frame_mean=float(np.mean([v['equal_bin_abs_residual_jump_mean'] for v in complete])) if complete else None))
    out=dict(schema='FROZEN_DOMAIN_THREE_LAYER_FUSION_WRITEBACK_CPU_REVIEW_V2',mask_registry_sha256=sha(a.mask_registry),raws=raws,methods=methods,
        rows=rows,pooled=pooled,seam_pooled=seam_pooled,primary_bbox_registered_pairs=primary_pairs,input_sha256=source_sha,native_receipt_sha256=sha(a.readouts/'receipt.json') if receipt else None,
        native_receipt_pooled_recheck=receipt_pool_check(receipt),target_manifest_sha256=sha(target_manifest) if target_contract else None,
        readout_role='BASELINE_TARGET_PREFLIGHT' if all(a.method_pattern.format(method=m,state='delete')==a.baseline_pattern.format(state='delete') for m in methods) else 'WRITEBACK_EVALUATION',
        executed_sources_sha256={Path(__file__).name:sha(Path(__file__)),'prepare_world_masks49.py':sha(Path(__file__).with_name('prepare_world_masks49.py'))},
        baseline='Previous n2+relativeMip+parent cellfit colour, replayed by native evaluator.',
        target_interpretation='Desired full-composited encoded-sRGB images from190 non-HELD sources; targets are not GT. '+('CPU source/fit-contract audit verified against the actual manifest SHA, with same-scene exposure limitations.' if audit_verified else 'Source audit missing or manifest SHA differs; contract audit remains pending.'),
        source_contract_audit=dict(path=str(audit_path),sha256=sha(audit_path),status=audit['status'],manifest_sha_matches=audit_verified,limitations=audit['limitations']) if audit else None,
        evidence_limits=['Same-scene development; original key5 mask bytes preserved, remaining44 masks fixed by original measurements only.',
            'Primary raw128 approximate user bbox and secondary world patch retain separate denominators.',
            'Unsupported edge is None, never zero; incomplete target bins retain registered denominator and are not silently pooled.',
            'Gaussian alpha/expected centre depth are not opaque first-hit measurements; fixed C2-ground ray categories have unknown hidden truth.',
            'HELD9 excludes current target-source/fitting when contract confirms, but inherited prior scene/assets were previously exposed; not independent end-to-end test.'],
        GPU_used=False,model_updates=0,whole_scene_adopted=False,seconds=time.time()-start)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(a.output),rows=len(rows),seconds=out['seconds'],sha256=sha(a.output))),flush=True)


if __name__=='__main__':main()
