"""Post-evaluation summaries; keep the original run artifacts and fit decisions intact."""
import argparse,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--r1',type=Path,required=True);p.add_argument('--r2',type=Path,required=True);a=p.parse_args()
    s=json.loads((a.r2/'summary.json').read_text());meta=json.loads((a.inputs/'manifest.json').read_text());methods=['fixed_same_verifier','image_plane','metric_plane','measured_refined']
    totals={str(act):{m:dict(fit=0,testable=0,validation_pass=0,validated_added_vs_legacy=0,validated_unsupported=0,geometry_proposals=0) for m in methods} for act in meta['actors']}
    rows=[];births=[];ns=[];colors=[];sources=[];unchanged=0;critical=[]
    for fid in meta['frame_order']:
        z=dict(np.load(a.r2/'frames'/f'{fid}.npz'));old=dict(np.load(a.r1/'frames'/f'{fid}.npz'));row={'frame':fid,'actors':{}}
        for m in methods[:-1]:
            for suffix in ['', '_xyz','_sources','_validation']:
                np.testing.assert_array_equal(z[m+suffix],old[m+suffix]);unchanged+=1
        for k,act in enumerate(meta['actors']):
            mask=z['actor']==k+1;entry={}
            for m in methods:
                v=z[m+'_validation'];entry[m]=dict(fit=int((mask&z[m]).sum()),testable=int((mask&z[m+'_testable']).sum()),validation_pass=int((mask&v).sum()),validated_added_vs_legacy=int((mask&v&~z['legacy']).sum()),validated_unsupported=int((mask&v&~z['supported']).sum()),geometry_proposals=int((mask&z[m+'_proposal_valid']).sum()))
                for key,value in entry[m].items():totals[str(act)][m][key]+=value
            row['actors'][str(act)]=entry
        rows.append(row)
        add=z['measured_refined_validation']&~z['fixed_same_verifier_validation']
        births.append(z['measured_refined_xyz'][add]);ns.append(z['measured_refined_normal'][add]);colors.append(z['measured_refined_rgb'][add]);sources.append(z['measured_refined_sources'][add])
        if fid=='0008102':
            for point in [[255,240],[256,219]]:
                i=np.flatnonzero((z['pixels']==point).all(1))[0];label=int(z['actor'][i])
                critical.append({'pixel':point,'actor_label_index':label,'actor_id':meta['actors'][label-1],
                    'methods':{m:{'proposal_exists':bool(z[m+'_proposal_valid'][i]),'fit_accepted':bool(z[m][i]),'validated':bool(z[m+'_validation'][i]),'selected_proposal_geometry_sources':int(z[m+'_geo_count'][i]),'candidate_local_xyz':z[m+'_xyz'][i].tolist()} for m in methods}})
    births=np.concatenate(births);ns=np.concatenate(ns);colors=np.concatenate(colors);sources=np.concatenate(sources)
    if len(births):
        _,ii=np.unique(np.floor(births/.1).astype(int),axis=0,return_index=True);births=births[ii];ns=ns[ii];colors=colors[ii];sources=sources[ii]
    np.savez_compressed(a.r2/'refined_validated_birth_proposals.npz',local_xyz=births,normal_local=ns,rgb=colors,source_indices=sources)
    worst={act:{m:sorted([{'frame':r['frame'],**r['actors'][act][m]} for r in rows],key=lambda q:q['validation_pass'])[:5] for m in methods} for act in totals}
    d=dict(actors=totals,per_frame=rows,critical=critical,worst_frames=worst,r1_r2_unchanged_arrays=unchanged,validated_new_position_voxels_10cm=len(births),scope='CPU proposal audit; validation is development consistency, not independent GT; no training labels installed',quality_accepted=False,human_verdict=None)
    (a.r2/'audit_detail.json').write_text(json.dumps(d,indent=2))
    print(json.dumps({'actors':totals,'critical':critical,'unchanged_arrays':unchanged,'birth_voxels':len(births)},indent=2))
if __name__=='__main__':main()
