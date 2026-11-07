"""Independent review addendum; never edits canonical NPZ/manifest/parity bytes."""
from pathlib import Path
import argparse
import json
import shutil
import tarfile
from world_surface_sampling import sha


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();run=args.run
    engineer=run/'engineer';candidate=engineer/'canonical_n2_mip_candidate_v2'
    manifest=candidate/'scene_manifest.json';asset=candidate/'new_road_canonical_candidate.npz'
    before=dict(scene_manifest=sha(manifest),asset=sha(asset),parity=sha(candidate/'parity/receipt.json'))
    assert before['scene_manifest']=='48ac926aafb1cb07ddd98c4ee6c9ffdc7c4501076a1cdbc7ffb99cc9736e84c7'
    assert before['asset']=='53de3cb31be234865995766eaeb11a5fced59160456bb139b08cdb3eb0031baf'
    target=candidate/'final_review.json'
    if target.exists():raise FileExistsError('Preserve independent review')
    source=run/'blue/fivekey_fixed_world_patch_comparison.json';data=json.loads(source.read_text())
    examples=[]
    for raw in (105,113,128,130,137):
        for state in ('keep','delete'):
            matches={row['method']:row for row in data['per_view'] if row['raw']==raw and row['state']==state}
            for edge in ('near_x','lateral_y'):
                base=matches['R0']['edges'][edge];aa=matches['AA']['edges'][edge]
                value=aa.get('equal_bin_abs_residual_jump_mean');ref=base.get('equal_bin_abs_residual_jump_mean')
                examples.append(dict(raw=raw,state=state,world_boundary=edge,
                    paired_tangent_bins=aa['paired_bins'],R0_residual_jump=ref,AA_residual_jump=value,
                    change=value-ref if value is not None and ref is not None else None,
                    status='unsupported' if value is None else ('regression' if value>ref else 'improved_or_equal')))
    review=dict(schema='CANONICAL_GEOMETRY_SUPPORT_REVIEW_ADDENDUM_V1',status='GEOMETRY_SUPPORT_RESEARCH_CANDIDATE',
        adoption_claim=False,canonical_asset_sha256=before['asset'],scene_manifest_sha256=before['scene_manifest'],
        replay_receipt_sha256=before['parity'],canonical_manifest_asset_and_parity_unmodified=True,
        blue_crossframe_seam_status='UNRESOLVED_WITH_CROSSFRAME_BOUNDARY_REGRESSIONS',
        blue_conclusion='Raw128 near-x improves, but raw105 near-x and raw113/130/137 lateral-y residual jumps regress; no general seam-resolved claim.',
        blue_primary_source=str(source),blue_primary_source_sha256=sha(source),
        blue_raw128_material_control_source=str(run/'blue/blue_public_summary.json'),
        blue_raw128_material_control_source_sha256=sha(run/'blue/blue_public_summary.json'),
        boundary_evidence=examples,
        material_selection='unchanged parent cellfit1000; blue18coef control not included',
        review_limits=['Geometry support only; the whole new-road scene is not adopted.',
            'Blue seam is not resolved across the fixed five-view world patch; boundary-specific regressions are preserved above.',
            'Red inferred pointset support/ordering evidence does not establish true hidden-wheel geometry or texture preservation.',
            'Exact replay proves delivery reproducibility, not scientific quality acceptance.'],
        new_GPU_forwards=0,new_optimizer_updates=0,original_parameter_updates=0)
    write(target,review)
    evidence=engineer/'final_review_evidence_v1';evidence.mkdir(exist_ok=False)
    for src,name in ((source,'fivekey_fixed_world_patch_comparison.json'),
        (run/'blue/blue_public_summary.json','blue_public_summary.json'),
        (run/'blue/fivekey_world_patch_masks_v1/registry.json','fivekey_registry.json'),
        (Path(__file__),'append_final_review.py')):shutil.copy2(src,evidence/name)
    after=dict(scene_manifest=sha(manifest),asset=sha(asset),parity=sha(candidate/'parity/receipt.json'));assert after==before
    files=[target]+[path for path in evidence.iterdir() if path.is_file()]
    inventory={str(path.relative_to(engineer)):dict(bytes=path.stat().st_size,sha256=sha(path)) for path in files}
    index=evidence/'review_addendum_manifest.json'
    write(index,dict(files=inventory,base_archive='canonical_delivery_v2.tar.gz',
        base_archive_sha256=sha(engineer/'canonical_delivery_v2.tar.gz'),frozen_before_after_hashes=after,
        status=review['status'],adoption_claim=False))
    archive=engineer/'final_review_addendum_v1.tar.gz'
    if archive.exists():raise FileExistsError('Preserve addendum archive')
    with tarfile.open(archive,'w:gz') as tar:
        for path in files+[index]:tar.add(path,arcname=str(path.relative_to(engineer)),recursive=False)
    print(json.dumps(dict(archive=str(archive),archive_sha256=sha(archive),review_sha256=sha(target),
        bytes=archive.stat().st_size,files=len(inventory),frozen_before_after_hashes=after)))


if __name__=='__main__':main()
