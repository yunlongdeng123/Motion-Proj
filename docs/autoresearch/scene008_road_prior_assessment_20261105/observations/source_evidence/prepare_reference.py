"""Build stable clean-road evidence for a5 AND a37 from all 199 real frames.

Reuses audited measured triangles/occlusion records, not their old 49-frame
colour eligibility. No model checkpoint is loaded or changed.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key] = '1'
import json, warnings, sys, time
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
FULL = ROOT/'r15_gn11_fullclip_cpu_20261026'
BUNDLE = ROOT/'r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json'

def linear(x):
    x = np.clip(x,0,1)
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)

def main():
    started = time.time(); out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    target = out/'road_reference_199.npz'
    if target.exists():
        raise FileExistsError('Keep old evidence; use a new run directory')
    m = json.loads(BUNDLE.read_text()); a = dict(np.load(Path(__file__).resolve().parent.parent/'inputs/expanded/evidence.npz'))
    assert np.array_equal(a['source_raw'],np.arange(199))
    valid = a['source_valid'].astype(bool)
    observed = linear(a['source_rgb']).astype(np.float32)
    luma = observed@np.array([.2126,.7152,.0722],np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        q75 = np.nanquantile(np.where(valid,luma,np.nan),.75,axis=0)
        upper = valid&(luma>=q75[None]-.025)&(luma<=q75[None]+.06)
        ref = np.nanmedian(np.where(upper[:,:,None],observed,np.nan),axis=0)
    ref = np.nan_to_num(ref,nan=0).astype(np.float32)
    agree = valid&(np.abs(observed-ref[None]).max(2)<.035)
    count = agree.sum(0)
    first = np.where(agree,np.arange(199)[:,None],999).min(0)
    last = np.where(agree,np.arange(199)[:,None],-1).max(0)
    supported = a['surface_supported']&(count>=3)&(last-first>=6)&(ref.min(1)>.012)&(ref.max(1)<.88)
    scale = m['meter_to_model']; xyz = a['model_xyz']
    distances = {}; inside = {}
    # World-space sweep plus 2m context includes under-car and nearby cast-shadow
    # receiver. This is a repair domain, never a shadow/ownership ground truth.
    for alias in [5,37]:
        d = np.full(len(xyz),np.inf)
        k = m['actors'].index(alias)
        for fid in m['frame_order']:
            box = m['frames'][fid]['selected_boxes_model'][k]
            local = (xyz-np.asarray(box['center']))@np.asarray(box['axes'])/scale
            gap = np.maximum(np.abs(local[:,:2])-np.asarray(box['half'])[:2]/scale,0)
            d = np.minimum(d,np.linalg.norm(gap,axis=1))
        distances[alias]=d; inside[alias]=d<=2.
    domain = inside[5]|inside[37]
    active = supported&domain
    keys = a['keys'].astype(int); lo = keys.min(0); shape = keys.max(0)-lo+1
    ij = keys-lo; chart = np.zeros((shape[1],shape[0]),bool)
    chart[ij[:,1],ij[:,0]] = active
    interior = binary_erosion(chart,iterations=1)[ij[:,1],ij[:,0]]
    # Boundary has no replacement authority; unsupported cells remain unknown.
    np.savez_compressed(target,keys=a['keys'],local_xyz=a['local_xyz'],model_xyz=xyz,
        world_xyz=a['world_xyz'],normal_local=a['normal_local'],normal_world=a['normal_world'],
        gaussian_sigma_m=a['gaussian_sigma_m'],reference_linear=ref,reference_srgb=np.where(ref<=.0031308,ref*12.92,1.055*np.maximum(ref,1e-10)**(1/2.4)-.055),
        active=active,route_interior=interior,supported=supported,domain=domain,
        domain_a5=inside[5],domain_a37=inside[37],distance_a5_m=distances[5],distance_a37_m=distances[37],
        agree_count=count,source_valid=valid,source_agree=agree,source_raw=a['source_raw'],
        first_clean_source=first,last_clean_source=last,source_rgb=a['source_rgb'],cell_m=a['cell_m'])
    report = dict(schema='A5_A37_REAL199_CLEAN_ROAD_V1',source_frames=199,source_raw_range=[0,198],
        build_frames=list(range(199)),evaluation_raw_range=[89,137],gradient_camera_raw_range=[89,137],
        all199_used_for_geometry_and_rgb=True,geometry_cells=len(xyz),cell_m=float(a['cell_m']),
        road_qualified=int(supported.sum()),domain_cells=int(domain.sum()),active_cells=int(active.sum()),
        route_interior_cells=int(interior.sum()),actors={str(k):dict(domain_cells=int(inside[k].sum()),
        qualified_cells=int((supported&inside[k]).sum()),interior_cells=int((interior&inside[k]).sum()),
        unknown_cells=int((inside[k]&~supported).sum()),geometric_source_samples=int(valid[:,inside[k]].sum()),
        clean_source_samples=int(agree[:,inside[k]].sum()),source_frames_with_geometric_evidence=int(valid[:,inside[k]].any(1).sum()),
        source_frames_with_clean_evidence=int(agree[:,inside[k]].any(1).sum())) for k in [5,37]},
        clean_rule='measured geometric witnesses; upper stable linear-radiance cluster q75[-.025,+.06], maxRGB agreement<.035, >=3 agreeing views, temporal span>=6 frames',
        roi_rule='a5/a37 49-frame world footprint sweep +2m receiver context; no new shadow layer',
        unknown_policy='no source or unstable colour => no repair authority; report separately',
        clean_reference_is_proxy=True,all_199_are_build_inputs=True,heldout_scene_claim=False,
        optimizer_updates=0,gpu_used=False,seconds=time.time()-started)
    (out/'reference_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__ == '__main__': main()
