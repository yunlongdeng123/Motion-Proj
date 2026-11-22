"""Actual cell-centre all-four appearance coverage on immutable geometry."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from atlas_canvas_cpu import minimal_vertex_scope,needed_shape


def bank(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['geometry','field','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--field-sha256',required=True);a=p.parse_args();start=time.time();assert not a.output.exists()
    assert sha(a.field)==a.field_sha256;g=bank(a.geometry);f=bank(a.field);H,W=f['permission_mask'].shape
    mask=g['cell_asset_mask'];uv=g['cell_uv_global'];assert uv.shape==(*mask.shape,2)
    finite=np.isfinite(uv).all(-1);safe=np.where(finite[...,None],uv,0);x=np.floor(safe[...,0]).astype(int);y=np.floor(safe[...,1]).astype(int)
    bounds=finite&(x>=0)&(y>=0)&(x+1<W)&(y+1<H);xc=np.clip(x,0,W-2);yc=np.clip(y,0,H-2)
    perm=np.ones(mask.shape,bool);kind=perm.copy();colour=perm.copy()
    for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
        perm&=f['permission_mask'][yc+dy,xc+dx];kind&=f['source_kind'][yc+dy,xc+dx]>0;colour&=np.isfinite(f['atlas_linear'][yc+dy,xc+dx]).all(-1)
    legal=bounds&perm&kind&colour;vertex=minimal_vertex_scope(mask);shape=needed_shape(g['uv_global'],vertex,(H,W))
    reasons=dict(nonfinite_uv=mask&~finite,outside_existing_canvas=mask&finite&~bounds,
        missing_permission=mask&bounds&~perm,missing_source_kind=mask&bounds&perm&~kind,nonfinite_source_RGB=mask&bounds&perm&kind&~colour)
    assert sum(int(v.sum()) for v in reasons.values())==int((mask&~legal).sum())
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'coverage.npz',cell_asset_mask=mask,parent_all_four_legal=legal,unknown_cells=mask&~legal,minimal_vertex_seed_scope=vertex,**reasons)
    # This is a scope copy only; certified/inferred geometry values remain exact.
    scoped=dict(g);scoped['minimal_asset_vertex_scope']=vertex
    np.savez_compressed(a.output/'geometry_source_scope.npz',**scoped)
    receipt=dict(schema='R10.expanded_source_actual_all_four_coverage.v1',complete=True,CPU_only=True,nativeF=0,nativeB=0,updates=0,
        geometry=pin(a.geometry),field=pin(a.field),asset_cells=int(mask.sum()),parent_reusable_asset_cells=int((mask&legal).sum()),
        unknown_asset_cells=int((mask&~legal).sum()),exclusive_reason_counts={k:int(v.sum()) for k,v in reasons.items()},
        parent_canvas_shape=[H,W],needed_canvas_shape=list(shape),uv_origin_global=[0,0],texel_spacing_m=.02,
        minimal_seed_vertices=int(vertex.sum()),geometry_scope_copy=pin(a.output/'geometry_source_scope.npz'),coverage=pin(a.output/'coverage.npz'),
        original_geometry_not_modified=True,source_sampling='all four neighbours at actual cell UV+.5, permission/source-kind/finite-RGB; no RGB colour threshold',
        decision='REUSE_PARENT_DIRECTLY' if (mask&legal).sum()==mask.sum() else 'UNKNOWN_REQUIRES_NEW_EXPLICIT_SOURCE_CANDIDATE',
        new_diffusion_calls=0,new_semantic_calls=0,geometry_or_source_accepted=False,Gaussian_fit_authorized=False,seconds=time.time()-start)
    (a.output/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
