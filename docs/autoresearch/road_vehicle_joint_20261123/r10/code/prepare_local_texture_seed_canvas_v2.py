"""Extend photographic texture into a separately supplied local geometry scope.

All previously accepted atlas RGB stays byte-exact. Parent field values are
Dirichlet/reference anchors, not newly relabelled same-position photographs.
This CPU candidate creates no Gaussian asset and authorizes no fitting.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
from atlas_canvas_cpu import embed,minimal_vertex_scope,needed_shape

PARENT_SHA='2eaf1569a97fb8ee6907696dfa28db91bb5f3072f74d37491007c87012e2c647'


def bank(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def pin(path):return dict(path=str(path),sha256=sha(path),bytes=Path(path).stat().st_size)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--r7',type=Path,required=True);p.add_argument('--parent-field',type=Path,required=True)
    p.add_argument('--parent-sha256',default=PARENT_SHA,help='Explicit approved parent pin; the default is the first accepted shared field')
    p.add_argument('--local-geometry',type=Path,required=True)
    p.add_argument('--candidate-mask-key',required=True,help='Explicit candidate scope, not an implicit validity fallback')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert sha(a.parent_field)==a.parent_sha256,'The explicitly pinned parent field changed'
    g=bank(a.r7/'shared_grid_geometry.npz');parent=bank(a.parent_field);local=bank(a.local_geometry)
    xyz=local['world_xyz'];uv=local['uv_global']
    scope=minimal_vertex_scope(local['cell_asset_mask']) if a.candidate_mask_key=='__asset_vertices__' else local[a.candidate_mask_key].astype(bool)
    assert xyz.shape==(*scope.shape,3) and uv.shape==(*scope.shape,2)
    assert np.isfinite(xyz[scope]).all() and np.isfinite(uv[scope]).all()
    integer=np.rint(uv).astype(np.int64);assert np.array_equal(uv[scope],integer[scope])
    original_shape=parent['permission_mask'].shape;canvas_shape=needed_shape(uv,scope,original_shape)
    parent=embed(parent,canvas_shape,original_shape);g=embed(g,canvas_shape,original_shape)
    H,W=canvas_shape
    inside=(integer[...,0]>=0)&(integer[...,1]>=0)&(integer[...,0]<W)&(integer[...,1]<H)
    assert inside[scope].all(),'Out-of-parent-grid extension needs a separate coordinate contract'
    iy,ix=integer[...,1][scope],integer[...,0][scope]
    assert len(np.unique(iy*W+ix))==len(ix),'Duplicate local UV coordinates'
    mask=np.zeros((H,W),bool);mask[iy,ix]=True
    extension=mask&~parent['permission_mask'];assert extension.any(),'No new unknown material scope'
    assert not a.output.exists();a.output.mkdir(parents=True);started=time.time()
    sys.path.insert(0,str(a.r7/'code'))
    from quilt_shared_appearance import quilt_arrays
    full_world=g['world_xyz'].copy();full_world[iy,ix]=xyz[scope]
    domain=parent['permission_mask']|extension
    geometry=dict(world_xyz=full_world,world_from_uv_xy=g['world_from_uv_xy'],
                  write_domain=domain,geometry_valid=domain)
    # quilt_arrays uses this mask only as an immutable numerical reference.
    # Its temporary label1 is NOT exported as real-photo provenance.
    observation=dict(canonical_linear_rgb=parent['atlas_linear'],known=parent['permission_mask'],
                     confidence=parent['permission_mask'].astype(np.float32))
    kwargs=dict(tile_size=24,overlap=6,seed=20261122,candidate_count=48,context=0,seam_dilate=0)
    manifest=a.r7/'texture_bank/texture_bank_manifest.json';gains=a.r7/'exposure_gain.json'
    quilt,trace=quilt_arrays(geometry,observation,manifest,gains,**kwargs)
    replay,_=quilt_arrays(geometry,observation,manifest,gains,replay_operations=trace['operations'],**kwargs)
    assert np.array_equal(quilt['atlas_seed_linear'],replay['atlas_seed_linear'])
    output={k:v.copy() for k,v in parent.items()}
    output['atlas_linear'][extension]=quilt['atlas_seed_linear'][extension]
    output['source_kind'][extension]=2
    output['seed_source_kind'][extension]=2
    output['permission_mask']=domain;output['write_domain']=domain.copy()
    output['confidence'][extension]=.25
    for key in ['teacher_adopted','teacher_support_count','teacher_directly_observed','teacher_inferred_field','effective_teacher_blend']:
        if key in output:output[key][extension]=0
    output['parent_conditioned_extension']=extension
    output['local_geometry_scope']=mask
    output['canvas_parent_shape']=np.asarray(original_shape,np.int64);output['canvas_shape']=np.asarray(canvas_shape,np.int64)
    output['canvas_uv_origin_global']=np.asarray([0,0],np.int64)
    assert output['real_locks'].tobytes()==parent['real_locks'].tobytes()
    assert output['atlas_linear'][~extension].tobytes()==parent['atlas_linear'][~extension].tobytes()
    assert np.isfinite(output['atlas_linear']).all()
    np.savez_compressed(a.output/'texture_seed.npz',**output)
    np.savez_compressed(a.output/'new_texture_lineage.npz',extension_mask=extension,
        mixture_tile_id=quilt['mixture_tile_id'],mixture_tile_weight=quilt['mixture_tile_weight'],
        source_raw_primary=quilt['source_raw_primary'],source_uv_Brown_primary=quilt['source_uv_Brown_primary'])
    trace['numerical_anchor_semantics']='immutable accepted parent field; its source1/2/3 provenance is inherited, not relabelled as new same-world real photographs'
    trace['new_region_source_kind']='similar-photograph texture inference conditioned on the accepted parent boundary, including any inherited teacher colour conditioning'
    write(a.output/'quilt_trace.json',trace)
    write(a.output/'receipt.json',dict(schema='R10.local_photographic_texture_seed_candidate.v1',complete=True,
        parent_field=pin(a.parent_field),local_geometry=pin(a.local_geometry),candidate_mask_key=a.candidate_mask_key,
        same_global_UV_canvas_embedding=dict(parent_shape=list(original_shape),canvas_shape=list(canvas_shape),origin_global=[0,0],old_array_values_preserved=True,new_canvas_rows_unknown_before_quilting=True,new_world_XYZ_only_from_candidate_geometry=True),
        producer=pin(__file__),canvas_helper=pin(Path(__file__).with_name('atlas_canvas_cpu.py')),
        texture_bank_manifest=pin(manifest),exposure=pin(gains),new_region_texels=int(extension.sum()),
        parent_RGB_outside_new_region_byteexact=True,parent_real_lock_mask_byteexact=True,
        deterministic_texture_replay=True,seed=pin(a.output/'texture_seed.npz'),
        lineages=pin(a.output/'new_texture_lineage.npz'),trace=pin(a.output/'quilt_trace.json'),
        geometry_acceptance='NOT asserted by this texture preparation',material_acceptance='PENDING joint teacher fusion and direct-surface review',
        parent_conditioning_is_not_pure_real_photo=True,Gaussian_fit_authorized=False,
        GPU_calls=0,new_diffusion_calls=0,new_semantic_calls=0,Gaussian_updates=0,seconds=time.time()-started))
    print(json.dumps(dict(complete=True,new_region_texels=int(extension.sum()),seed_sha256=sha(a.output/'texture_seed.npz'))))


if __name__=='__main__':main()
