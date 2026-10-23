"""Reproject actual sources and verify R5 immutability, new colors, split provenance and fixed capacity."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from scipy.ndimage import label
from prepare_v41_revision import native_rgb,witness_at,sample_rgb


def main():
    p=argparse.ArgumentParser()
    for key in ('dense','engine','output'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();sys.path.insert(0,str(args.engine))
    from utils.v41_sources import EVIDENCE_VERSION,SOURCE_POLICY,source_consensus,first_two,validate_label_sources
    m=json.loads((args.dense/'manifest.json').read_text());prior=Path(m['derived_from'])
    meta=json.loads(Path(m['source_bundle']).read_text())
    assert m['version']==EVIDENCE_VERSION and m['policy']['source_expansion']==SOURCE_POLICY
    evidence=[];rgb=[]
    for fid in m['frame_order']:
        evidence_path=args.dense/'frames'/f'{fid}_evidence.npz'
        assert evidence_path.read_bytes()==(prior/'frames'/evidence_path.name).read_bytes()
        with np.load(evidence_path,allow_pickle=False) as z:evidence.append((z['box_clear_witness'].copy(),z['measured_patch_depth_m'].copy()))
        rgb.append(native_rgb(meta['frames'][fid]['original_rgb']))
    changed={'deletion_source_ids','deletion_source_uv','deletion_valid','deletion_domains','deletion_target_rgb'}
    checks=0;added_total=0;legacy_total=0;actor_totals=np.zeros(5,np.int64)
    for i,fid in enumerate(m['frame_order']):
        with np.load(args.dense/'frames'/f'{fid}.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
        with np.load(prior/'frames'/f'{fid}.npz',allow_pickle=False) as z:old={k:z[k] for k in z.files}
        for k in old:
            if k not in changed:np.testing.assert_array_equal(data[k],old[k],err_msg=k)
        legacy=old['deletion_valid'];added=data['deletion_label_kind']==2
        np.testing.assert_array_equal(data['deletion_legacy_valid'],legacy)
        np.testing.assert_array_equal(data['deletion_target_rgb'][:,legacy],old['deletion_target_rgb'][:,legacy])
        np.testing.assert_array_equal(data['deletion_source_ids'][legacy],old['deletion_source_ids'][legacy])
        np.testing.assert_array_equal(data['deletion_valid'],legacy|added)
        assert not (added & (~old['deletion_candidate']|legacy)).any()
        validate_label_sources(data,i)
        with np.load(args.dense/'consensus'/f'{fid}.npz',allow_pickle=False) as z:c={k:z[k] for k in z.files}
        candidates=old['deletion_candidate']&~legacy;yy,xx=np.where(candidates)
        np.testing.assert_array_equal(c['candidate_pixels'],np.flatnonzero(candidates))
        frame=meta['frames'][fid];c2w=np.linalg.inv(frame['world_to_camera'])
        rays=np.c_[xx+.5,yy+.5,np.ones(len(xx))] @ np.linalg.inv(frame['intrinsic']).T
        world=(rays*old['fixed_surface_depth_m'][candidates,None]) @ c2w[:3,:3].T+c2w[:3,3]
        fresh=np.full_like(c['colors'],np.nan)
        for j,source in enumerate(m['frame_order']):
            if j==i:
                assert not np.isfinite(c['colors'][j]).any();continue
            clear,depth=evidence[j];valid,uv=witness_at(world,meta['frames'][source],clear,depth)
            np.testing.assert_array_equal(np.isfinite(c['colors'][j,:,0]),valid)
            np.testing.assert_allclose(c['source_uv'][j,valid],uv[valid],atol=2e-5)
            fresh[j,valid]=sample_rgb(rgb[j],uv[valid]);checks+=int(valid.sum())
        np.testing.assert_allclose(fresh,c['colors'],atol=1e-6,equal_nan=True)
        result=source_consensus(fresh,i)
        for k,v in result.items():np.testing.assert_allclose(v,c[k],atol=1e-6)
        proposed=np.zeros((320,480),bool);proposed[candidates]=result['accepted']
        expected=np.zeros_like(proposed)
        for a in range(1,6):
            parts,n=label(proposed&(old['actor_labels']==a),structure=np.ones((3,3),bool))
            sizes=np.bincount(parts.ravel());keep=sizes>=9;keep[0]=False;expected|=keep[parts]
        np.testing.assert_array_equal(added,expected)
        np.testing.assert_array_equal(c['accepted_after_spatial'],added[candidates])
        np.testing.assert_allclose(data['deletion_target_rgb'][:,added].T,result['target'][added[candidates]],atol=1e-6)
        np.testing.assert_array_equal(data['deletion_source_ids'][added],first_two(result['fit_source_mask'])[added[candidates]])
        np.testing.assert_array_equal(data['deletion_validation_source_ids'][added],first_two(result['validation_source_mask'])[added[candidates]])
        assert np.array_equal(data['deletion_domains'],np.stack([data['deletion_valid']&(old['actor_labels']==a) for a in range(1,6)]))
        actor_totals+=data['deletion_domains'].sum((1,2));legacy_total+=int(legacy.sum());added_total+=int(added.sum())
        print(json.dumps(dict(frame=fid,added=int(added.sum()))),flush=True)
    result=dict(status='PASS_ACTUAL49_MULTISOURCE',frames=49,source_checks=checks,
        legacy_labels=legacy_total,added_labels=added_total,total_labels=legacy_total+added_total,
        actors=dict(zip(map(str,(5,26,18,6,37)),map(int,actor_totals))),
        legacy_colors_unchanged=True,visible_road_masks_unchanged=True,vehicle_protection_unchanged=True,
        fixed_geometry_and_R_eligibility_unchanged=True,validation_colors_used_for_target=False,
        current_vehicle_RGB_used_for_target=False,GPU_used=False,optimizer_updates=0,
        independent_test_set=False,quality_accepted=False)
    with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
