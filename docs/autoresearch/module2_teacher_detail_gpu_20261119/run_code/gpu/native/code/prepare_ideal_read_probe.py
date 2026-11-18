"""CPU-only ideal world fields on the frozen true chart; no renderer imports."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,sys,time
import numpy as np

B=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
Q=B/'road_teacher_detail_cpu_20261119_r1';P=B/'road_multiscale_material_20261118_r1'
C=B/'road_capacity_cpu_20261117_r1'
O=Q/'gpu_20261119/ideal_read_probe'
sys.path.insert(0,str(C/'ideal_code'));sys.path.insert(0,str(Q/'pixel_render/code'))
from ideal_world_teacher import evaluate_world,sampling_qualification
from pixel_quadrature_stream import iter_pixel_chunks

def write(p,value):
    Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    t0=time.time();O.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((C/'ideal_targets_r2/field_config.json').read_text())
    cases=[dict(name='asphalt',asphalt='lowfreq',marking=None)]+[c for c in cfg['cases'] if c['name'] in ('marking_w120','marking_w240')]
    base=B/'road_asset_20261116_r1/phase2/formal_r1/asset/base_atlas.npz'
    with np.load(base) as z:xyz=z['surface_world_xyz'].copy()
    with np.load(Q/'permission/final_r1/teacher_evidence.npz') as z:
        valid=z['common_material_valid'].astype(np.float64);active=z['active_world_rows'].copy();eligible=z['common_write_active'].copy()
    H,W=valid.shape
    # This target is an independently specified world function, never native GS self-render.
    fields=[evaluate_world(xyz,c,cfg)[0] for c in cases]
    packed=np.concatenate(fields,-1).reshape(-1,9);v=valid.ravel()
    np.savez_compressed(O/'ideal_world_nodes.npz',rgb=np.stack(fields),common_material_valid=valid.astype(bool),surface_world_xyz=xyz)
    def query(uv,weights,geometry=None):
        u=np.asarray(uv,np.float64);n,q=u.shape[:2];lo=np.floor(u).astype(np.int64);frac=u-lo
        w=np.broadcast_to(weights,(n,q)).copy()
        if geometry is not None:w*=geometry
        out=np.zeros((n,9));mass=np.zeros(n)
        for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
            x=lo[...,0]+dx;y=lo[...,1]+dy;inside=(x>=0)&(x<W)&(y>=0)&(y<H)
            index=np.clip(y,0,H-1)*W+np.clip(x,0,W-1)
            b=(frac[...,0] if dx else 1-frac[...,0])*(frac[...,1] if dy else 1-frac[...,1])
            point=w*b*inside*v[index]
            out+=np.einsum('nq,nqc->nc',point,packed[index],optimize=True);mass+=point.sum(1)
        return np.divide(out,mass[:,None],out=np.zeros_like(out),where=mass[:,None]>1e-12),mass
    plan=Q/'permission/avg_plan_gh31_r1'
    uv=np.load(plan/'sample_uv_f64.npy',mmap_mode='r');gv=np.load(plan/'geometric_sample_valid.npy',mmap_mode='r')
    with np.load(plan/'geometry_plan.npz') as z:
        weights=z['weights'].copy();ids=z['query_world_rows'].copy();local=z['active_local_indices'].copy()
    if not np.array_equal(active[local],ids) or not np.array_equal(active[eligible],ids):raise RuntimeError('Ideal scope row mapping mismatch')
    result=np.zeros((len(ids),9));mass=np.zeros(len(ids));step=512
    def chunk(start):
        end=min(start+step,len(ids));a,b=query(uv[start:end],weights,gv[start:end]);return start,end,a,b
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start,end,a,b in pool.map(chunk,range(0,len(ids),step)):
            result[start:end]=a;mass[start:end]=b
    np.savez_compressed(O/'avg_rows.npz',rgb=result.reshape(-1,3,3).transpose(1,0,2).astype(np.float32),mass=mass,
        query_world_rows=ids,active_local_indices=local,active_world_rows=active,eligible=eligible)
    print('AVG_READY',time.time()-t0,flush=True)
    receipts=[]
    for raw in (86,110,134,179,198):
        with np.load(Q/'pixel_render/maps'/f'{raw:03d}.npz') as z:m={k:z[k] for k in z.files}
        use=np.flatnonzero(m['canonical_mesh_hit_valid'].ravel());pixel=np.zeros((320*480,9));pmass=np.zeros(320*480)
        for selected,samples,w,meta in iter_pixel_chunks(m['center_uv'],m['pixel_uv_jacobian'],use,density=2.,point_budget=131072):
            a,b=query(samples,w);pixel[selected]=a;pmass[selected]=b
        target=np.concatenate([evaluate_world(m['surface_world_xyz'],c,cfg)[0] for c in cases],-1)
        # Source-center sampling isolates the fixed 2cm chart interpolation from the pixel box footprint.
        center,cmass=query(m['center_uv'].reshape(-1,1,2),np.ones(1))
        # Convert chart UV Jacobian into true world XY derivatives on the canonical hit triangle.
        qualification={}
        # Cached independent ideal geometry retains the verified world-XY ray Jacobian.
        with np.load(C/'ideal_targets_r2/geometry'/f'{raw:03d}.npz') as g:
            for k in g.files:
                if 'jacobian' in k.lower():jac=g[k].copy();break
            else:jac=None
        for c in cases[1:]:
            if jac is not None:
                q=sampling_qualification(m['surface_world_xyz'],jac,c,cfg)
                for k,a in q.items():qualification[c['name']+'_'+k]=a
        np.savez_compressed(O/f'{raw:03d}_pixel.npz',rgb=pixel.reshape(320,480,3,3).transpose(2,0,1,3).astype(np.float32),mass=pmass.reshape(320,480),
            center_rgb=center.reshape(320,480,3,3).transpose(2,0,1,3).astype(np.float32),center_mass=cmass.reshape(320,480),
            independent_world_center_rgb=target.reshape(320,480,3,3).transpose(2,0,1,3).astype(np.float32),
            D=m['fixed_evaluation_domain'],canonical_mesh_hit_valid=m['canonical_mesh_hit_valid'],**qualification)
        receipts.append(dict(raw=raw,canonical_hits=len(use),material_valid_hits=int((pmass[use]>1e-12).sum())))
        print('PIXEL_READY',raw,time.time()-t0,flush=True)
    write(O/'prepared.json',dict(schema='ideal_native_read_probe_preparation.v1',completed=True,cases=[c['name'] for c in cases],
        source_function=str(C/'ideal_code/ideal_world_teacher.py'),source_config=str(C/'ideal_targets_r2/field_config.json'),
        atlas_geometry=str(base),scope_rows=len(ids),frames=receipts,seconds=time.time()-t0,optimizer_updates=0,
        GPU_executed=False,native_calls=0,average='true world covariance GH31 -> exact cached chart UV -> bilinear 2cm ideal field, common valid normalization',
        pixel='true G_s hit and local camera UV Jacobian -> uncapped anisotropic density2 pixel box, common valid normalization',
        reference='Independent continuous world function at true G_s pixel centers; source-grid and pixel-footprint convention are explicitly separate',
        no_claim='No optimization/capability upper bound or actual teacher-detail quality inference'))

if __name__=='__main__':main()
