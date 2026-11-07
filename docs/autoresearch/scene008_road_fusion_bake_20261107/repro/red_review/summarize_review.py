"""Summarize immutable three-layer results without a combined success label."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stat(v):
    v=np.asarray(v,float)
    return dict(n=len(v),mean=float(v.mean()) if len(v) else None,p95=float(np.percentile(v,95)) if len(v) else None,min=float(v.min()) if len(v) else None,max=float(v.max()) if len(v) else None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    report=json.loads(a.review.read_text());rows=report['rows'];methods=report['methods'];raw128=[];global_metrics=[];boundaries=[]
    for r in rows:
        if r['raw']!=128:continue
        name='primary_bbox_original_road';t=r.get('layer1_target',{})
        raw128.append(dict(raw=128,state=r['state'],method=r['method'],original_road=r['layer3_original_observations'][name],
            render_near=r['primary_bbox_boundaries'],target_fidelity=t.get('fidelity_to_original_visible',{}).get(name),
            replace_only_target_fidelity=t.get('replace_only_fidelity_to_original',{}).get(name),target_composition=t.get('domain_composition',{}).get(name),
            target_near=t.get('primary_bbox_boundaries',{}).get('near_x'),
            writeback_to_target=r.get('layer2_writeback_to_target',{}).get('error',{}).get(name),
            replace_only_writeback_to_target=r.get('layer2_writeback_to_target',{}).get('replace_only_error',{}).get(name)))
    scope='ALL49' if report['raws']==list(range(89,138)) else f"SELECTED_{len(report['raws'])}VIEWS"
    index={(r['state'],r['method'],r['subset'],r['domain'],r['variant']):r for r in report['pooled']}
    for method in methods:
        for state in ('keep','delete'):
            for domain in ('original_road','protected','a6_visible','a37_visible'):
                key=(state,method,scope,domain);b=index.get(key+('baseline',));c=index.get(key+('candidate',))
                if not b or not c:continue
                per=[dict(raw=r['raw'],delta=r['layer3_original_observations'][domain]['mean_delta']) for r in rows if r['state']==state and r['method']==method and domain in r['layer3_original_observations']]
                global_metrics.append(dict(state=state,method=method,domain=domain,count=c['count'],baseline=b['mean'],candidate=c['mean'],delta=c['mean']-b['mean'],
                    worse_frames=sum(r['delta']>0 for r in per),equal_frames=sum(r['delta']==0 for r in per),delta_distribution=stat([r['delta'] for r in per]),worst_frames=sorted(per,key=lambda x:x['delta'],reverse=True)[:5]))
            for edge in ('near_x','lateral_y','far_x'):
                changes=[];before=[];after=[];supported=0
                for r in rows:
                    if r['method']!=method or r['state']!=state:continue
                    b=r['world_boundaries']['baseline'][edge];c=r['world_boundaries']['candidate'][edge]
                    if b['status']=='UNSUPPORTED':continue
                    assert b['status']==c['status']=='COMPLETE';supported+=1
                    bi={x['tangent_bin']:x for x in b['bins']}
                    for v in c['bins']:
                        old=bi[v['tangent_bin']]['abs_channel_jump'];new=v['abs_channel_jump'];before.append(old);after.append(new)
                        changes.append(dict(raw=r['raw'],tangent_bin=v['tangent_bin'],minus_count=v['minus_count'],plus_count=v['plus_count'],baseline=old,candidate=new,delta=new-old))
                boundaries.append(dict(state=state,method=method,edge=edge,supported_views=supported,unsupported_views=len(report['raws'])-supported,paired_bins=len(changes),
                    baseline_equal_bin_mean=float(np.mean(before)) if before else None,candidate_equal_bin_mean=float(np.mean(after)) if after else None,
                    delta_equal_bin_mean=float(np.mean(after)-np.mean(before)) if before else None,
                    worse_bins=sum(r['delta']>0 for r in changes),improved_bins=sum(r['delta']<0 for r in changes),equal_bins=sum(r['delta']==0 for r in changes),
                    delta_distribution=stat([r['delta'] for r in changes]),worst_bins=sorted(changes,key=lambda x:x['delta'],reverse=True)[:5]))
    direct_fusion=[];ri={(r['raw'],r['state'],r['method']):r for r in rows}
    if set(('direct','fusion')).issubset(methods):
        for state in ('keep','delete'):
            for edge in ('near_x','lateral_y'):
                diffs=[]
                for raw in report['raws']:
                    d=ri[raw,state,'direct']['world_boundaries']['candidate'][edge];f=ri[raw,state,'fusion']['world_boundaries']['candidate'][edge]
                    if d['status']=='UNSUPPORTED':continue
                    di={v['tangent_bin']:v for v in d['bins']}
                    for b in f['bins']:diffs.append(b['abs_channel_jump']-di[b['tangent_bin']]['abs_channel_jump'])
                direct_fusion.append(dict(state=state,edge=edge,fusion_minus_direct=stat(diffs),fusion_worse_bins=sum(v>0 for v in diffs),fusion_better_bins=sum(v<0 for v in diffs)))
    float_checks=[v for r in rows for v in r['saved_float_invariants'].values() if v['status']!='NOT_SAVED_IN_FLOAT_NPZ']
    receipt_checks=[r['native_receipt'] for r in rows if r['native_receipt']['status']=='RECEIPT_HASH_COMPARISON']
    red=[dict(raw=r['raw'],method=r['method'],**r['fixed_red_rays']) for r in rows if 'fixed_red_rays' in r]
    out=dict(schema='INDEPENDENT_FULL49_FUSION_WRITEBACK_REVIEW_SUMMARY_V1',source_review=str(a.review),source_review_sha256=sha(a.review),
        scope=scope,raw128_primary=raw128,all49_original_domain_metrics=global_metrics,all49_world_seam=boundaries,fusion_vs_direct_world_seam=direct_fusion,
        frozen_checks=dict(saved_full_float_field_checks=len(float_checks),saved_float_different=sum(v['status']=='DIFFERENT' for v in float_checks),
            receipt_state_variant_records=len(receipt_checks),receipt_all5geometry_equal_records=sum(all(r['fields_equal'].values()) for r in receipt_checks),
            vehicle_row_hash_equal_records=sum(r['vehicle_row_hash_equal'] for r in receipt_checks),
            asset_premultiplied_RGB_exact_records=sum(bool(r['frozen_asset_premultiplied_RGB_exact']) for r in receipt_checks),
            a37_premultiplied_RGB_exact_records=sum(bool(r['frozen_a37_premultiplied_RGB_exact']) for r in receipt_checks),
            receipt_pooled_recheck=report['native_receipt_pooled_recheck']),
        fixed_red_rays=red,adoption_claim=False,combined_pass_label=False,GPU_used=False,
        interpretation='Target fidelity, native writeback fidelity, original visible-domain protection and fixed-ray preservation are separate. No hidden GT or scene adoption inferred.')
    a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(a.output),sha256=sha(a.output),raw128=[dict(state=r['state'],method=r['method'],road=r['original_road'],near=r['render_near']['candidate']['near_x']['equal_bin_abs_residual_jump_mean']) for r in raw128],
        all49_metrics=[r for r in global_metrics if r['domain'] in ('original_road','protected')],frozen=out['frozen_checks'])),flush=True)


if __name__=='__main__':main()
