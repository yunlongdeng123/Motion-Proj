"""CPU export of a self-contained active new-road research candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
RUN=CAM/'scene008_road_r0_v44_20261107_r1'
V44=CAM/'scene008_v44_rgb_asset_20261107_r1'
SOURCE_SHA='6112be03108b5a9b1a6deb2f605ef850dcd8ba5018346fa822993177815807a2'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as s:
        for b in iter(lambda:s.read(4194304),b''):h.update(b)
    return h.hexdigest()


def ahash(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes());return h.hexdigest()


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def write(path,v):Path(path).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def linear(x):return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)


def physical(a,colours):
    scale=float(a['meter_to_model']);n=a['normal_world']@a['world_to_model'][:3,:3].T/scale
    n=n/np.linalg.norm(n,axis=1)[:,None]
    q=np.c_[1+n[:,2],-n[:,1],n[:,0],np.zeros(len(n))];q/=np.linalg.norm(q,axis=1)[:,None]
    return dict(xyz=a['model_xyz'].astype(np.float32),scale=(a['gaussian_sigma_m']*scale).astype(np.float32),
        rotation=q.astype(np.float32),colour=colours.astype(np.float32),
        opacity_logit=np.full((len(colours),1),np.log(.99/.01),np.float32))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=RUN/'surface/frontfill_v2/road_reference_199.npz')
    parser.add_argument('--material',type=Path,required=True)
    parser.add_argument('--expected-material-sha',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    assert sha(args.source)==SOURCE_SHA and sha(args.material)==args.expected_material_sha
    a=load(args.source);m=load(args.material)
    assert str(m['asset_sha256'])==SOURCE_SHA
    support=m['support_active'].astype(bool);assert support.sum()==10098 and (~support).sum()==4563
    parent=m['extension_ids'][support];material_parent=np.flatnonzero(support)
    colours=m['colours_srgb'][support].copy();assert colours.shape==(10098,3) and np.isfinite(colours).all()
    assert np.array_equal(m['colours_srgb'][~support],a['reference_srgb'][m['extension_ids'][~support]])
    per_cell=('keys','local_xyz','world_xyz','model_xyz','normal_local','normal_world','gaussian_sigma_m',
        'geometry_measured_support','actual_return_support_count','support_pool_ids','support_distances_m',
        'support_height_residual_m','geometry_confidence','candidate_origin')
    payload={key:a[key][parent].copy() for key in per_cell}
    payload.update(active=np.ones(10098,bool),real_active=np.ones(10098,bool),
        reference_srgb=colours,reference_linear=linear(colours).astype(np.float32),rgb=colours.copy(),
        opacity=np.full(10098,.99,np.float32),opacity_logit=np.full((10098,1),np.log(.99/.01),np.float32),
        original_source_proxy_srgb=a['reference_srgb'][parent].copy(),
        original_source_proxy_linear=a['reference_linear'][parent].copy(),
        source_rgb=a['source_rgb'][:,parent].copy(),source_valid=a['source_valid'][:,parent].copy(),
        source_agree=a['source_agree'][:,parent].copy(),source_raw=a['source_raw'].copy(),
        source_valid_count=a['source_valid_count'][parent].copy(),agree_count=a['agree_count'][parent].copy(),
        source_valid_count49=a['source_valid'][89:138,parent].sum(0).astype(np.int16),
        source_photometric_confidence=a['photometric_confidence'][parent].copy(),
        original_appearance_provenance=a['appearance_provenance'][parent].copy(),
        appearance_provenance=np.full(10098,2,np.uint8),source_parent_cell_ids=parent,
        material_parent_extension_local_ids=material_parent,
        coordinate_anchor=a['coordinate_anchor'].copy(),world_to_model=a['world_to_model'].copy(),
        meter_to_model=a['meter_to_model'].copy(),cell_m=a['cell_m'].copy(),
        strip_lateral_bounds_m=np.array([-3.3500001571235396,2.7100001349769953]),
        source_asset_sha256=np.array(SOURCE_SHA),material_sha256=np.array(args.expected_material_sha),
        material_role=np.array('FITTED_VISIBLE_ROAD_MATERIAL_NOT_HIDDEN_RGB_TRUTH'),
        candidate_status=np.array('RESEARCH_CANDIDATE_NOT_ADOPTED'))
    for key in ('eligible','observed_mass_by_state','observed_frame_count_by_state'):
        if key in m:payload['material_'+key]=m[key][:,support].copy() if m[key].ndim==2 else m[key][support].copy()
    if 'metadata_json' in m:payload['parent_material_metadata_json']=m['metadata_json'].copy()
    target=args.output/'new_road_canonical_candidate.npz';np.savez_compressed(target,**payload)
    b=load(target)
    checks={key:ahash(b[key])==ahash(a[key][parent]) for key in per_cell}
    checks['fitted_colours_exact']=ahash(b['reference_srgb'])==ahash(colours)
    checks['real_source_proxy_separate_exact']=ahash(b['original_source_proxy_srgb'])==ahash(a['reference_srgb'][parent])
    checks['source_valid_exact']=ahash(b['source_valid'])==ahash(a['source_valid'][:,parent])
    checks['source_RGB_exact']=ahash(b['source_rgb'])==ahash(a['source_rgb'][:,parent])
    expected={key:a[key][parent] if np.asarray(a[key]).shape and len(a[key])==len(a['keys']) else a[key]
              for key in ('model_xyz','normal_world','gaussian_sigma_m','meter_to_model','world_to_model')}
    before=physical(expected,colours);after=physical(b,b['reference_srgb'])
    tuple_checks={key:ahash(value)==ahash(after[key]) for key,value in before.items()}
    assert all(checks.values()) and all(tuple_checks.values())
    assert not np.isnan(b['source_rgb'][b['source_valid']]).any()
    verification=dict(status='PASS_CPU_BAKED_EXPORT_RELOAD',rows=10098,active=10098,removed_inactive=4563,
        immutable_geometry_and_provenance_checks=checks,physical_fp32_tuple_checks=tuple_checks,
        physical_fp32_tuple_sha256={k:ahash(v) for k,v in after.items()},
        asset_sha256=sha(target),CPU_only=True,GPU_used=False,adoption_claim=False)
    write(args.output/'cpu_export_verification.json',verification)
    contract=json.loads((V44/'contract/input_contract.json').read_text())
    receipt=json.loads((V44/'results/full49_r1/v44/receipt.json').read_text())['runtime_receipt']
    dependency_paths=set(receipt['input_sha256'])|set(receipt['binding']['parent_identity'])
    dependency_paths|={str(RUN/'source/r0_runtime.py'),str(V44/'source/v44_runtime.py'),str(V44/'contract/input_contract.json')}
    dependencies={}
    for path in sorted(dependency_paths):
        p=Path(path);info=p.stat();digest=sha(p)
        if path in receipt['input_sha256']:assert digest==receipt['input_sha256'][path],path
        dependencies[path]=dict(path=path,bytes=info.st_size,mtime_ns=info.st_mtime_ns,sha256=digest)
    shutil.copy2(Path(__file__).parent/'baked_road_runtime.py',args.output/'baked_road_runtime.py')
    shutil.copy2(Path(__file__),args.output/'export_baked_candidate.py')
    if (Path(__file__).parent/'verify_baked_parity.py').exists():
        shutil.copy2(Path(__file__).parent/'verify_baked_parity.py',args.output/'verify_baked_parity.py')
    manifest=dict(schema='R0_PINNED_BAKED_SCENE_RESEARCH_CANDIDATE_V1',status='RESEARCH_CANDIDATE_NOT_ADOPTED',
        new_road=dict(file=target.name,sha256=sha(target),rows=10098,all_active=True,
            opacity_nominal=.99,colour_role='fitted persistent visible-road material; separate original source proxy'),
        wrapper=dict(file='baked_road_runtime.py',sha256=sha(args.output/'baked_road_runtime.py')),
        exporter_source_sha256=sha(__file__),GPU_parity_receipt='parity/receipt.json',
        r0_runtime_path=str(RUN/'source/r0_runtime.py'),v44_contract_path=str(V44/'contract/input_contract.json'),
        frozen_v44_binding_sha256=receipt['binding_sha256'],frozen_v44_binding=receipt['binding'],
        runtime_dependencies=dependencies,
        lineage=dict(source_asset=str(args.source),source_asset_sha256=SOURCE_SHA,
            material=str(args.material),material_sha256=args.expected_material_sha,
            policy='Only active original world-strip rows retained; all source geometry and row order retained; fitted colour baked once'),
        shared_scene_state=['C2','persistent S/F','SurfaceLOD','native static','old road','old gates','environment','cameras','trajectories','new road'],
        edit_operator='physical a37 existence only; keep/delete do not alter road or vehicle material/geometry',
        known_unresolved=['raw130 inferred vehicle W versus road occlusion remains separately under review',
            'No general claim of complete vehicle preservation or recovered hidden texture',
            'V2 proxy used all199; BUILD40 material fitting is not blinded road-prior evaluation'],
        CPU_export_verification='cpu_export_verification.json',adoption_claim=False)
    write(args.output/'scene_manifest.json',manifest)
    write(args.output/'asset_interface.json',dict(status=manifest['status'],asset=target.name,sha256=sha(target),
        arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in payload.items()},
        appearance_2='fitted real-visible-road material, not independently observed hidden RGB',
        source_rgb_role='unaltered geometrically-qualified original real FRONT observations; invalid samples remain unavailable NaN',
        geometry_role='LiDAR-supported interpolated road surface, not per-cell independent LiDAR measurements'))
    assert sha(args.source)==SOURCE_SHA and sha(args.material)==args.expected_material_sha
    print(json.dumps(dict(asset=str(target),sha256=sha(target),manifest=str(args.output/'scene_manifest.json'),dependencies=len(dependencies),physical_tuple=tuple_checks)),flush=True)


if __name__=='__main__':main()
