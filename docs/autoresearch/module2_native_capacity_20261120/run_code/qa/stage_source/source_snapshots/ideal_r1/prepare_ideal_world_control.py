"""CPU ideal targets from an independent world function and cached native weights.

The teacher is NOT the GS render of assigned ideal colors. Its variable color is
the analytic world field integrated over a native-pixel footprint on actual G_s.
readonly+We*ideal makes this a conditional color-capacity target, not hidden GT.
"""
from pathlib import Path
import argparse, importlib.util, sys, time, json
import numpy as np
from source_world_split import sha, load_keys, Q, HIGH, NEW

CAMPAIGNS=Q.parent
CANONICAL=CAMPAIGNS/'road_asset_20261116_r1/phase2/formal_r1/asset'
IDEAL=CAMPAIGNS/'road_capacity_cpu_20261117_r1'

def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

def triangle_world(uv,surface,supported):
    """Actual canonical diagonal topology, float64 UV -> world and local Jacobian."""
    uv=np.asarray(uv,np.float64);shape=uv.shape[:-1];flat=uv.reshape(-1,2);H,W=surface.shape[:2]
    finite=np.isfinite(flat).all(1);clean=np.where(finite[:,None],flat,0.)
    inside=finite&(clean[:,0]>=0)&(clean[:,0]<=W-1)&(clean[:,1]>=0)&(clean[:,1]<=H-1)
    lo=np.floor(clean).astype(np.int64);x=np.clip(lo[:,0],0,W-2);y=np.clip(lo[:,1],0,H-2)
    f=clean-np.stack([x,y],-1);u,v=f.T;upper=u>=v
    a=surface[y,x];b=np.where(upper[:,None],surface[y,x+1],surface[y+1,x+1]);c=np.where(upper[:,None],surface[y+1,x+1],surface[y+1,x])
    w=np.where(upper[:,None],np.stack([1-u,u-v,v],-1),np.stack([1-v,u,v-u],-1))
    xyz=a*w[:,0,None]+b*w[:,1,None]+c*w[:,2,None]
    ok=inside&supported[y,x]&np.where(upper,supported[y,x+1],supported[y+1,x+1])&np.where(upper,supported[y+1,x+1],supported[y+1,x])
    du=np.where(upper[:,None],b-a,b-c);dv=np.where(upper[:,None],c-b,c-a)
    jac=np.stack([du,dv],-1)
    return xyz.reshape(*shape,3),ok.reshape(shape),jac.reshape(*shape,3,2)

def bilinear_valid(uv,valid):
    uv=np.asarray(uv,np.float64);lo=np.floor(uv).astype(np.int64);f=uv-lo;H,W=valid.shape
    mass=np.zeros(uv.shape[:-1],np.float64)
    for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
        x=lo[...,0]+dx;y=lo[...,1]+dy;inside=(x>=0)&(x<W)&(y>=0)&(y<H)
        w=(f[...,0] if dx else 1-f[...,0])*(f[...,1] if dy else 1-f[...,1])
        mass+=w*inside*valid[np.clip(y,0,H-1),np.clip(x,0,W-1)]
    return mass

def compose_conditional_target(readonly,eligible_weight,ideal_rgb,valid,baseline):
    """CHW preclamp target, explicit fallback. Fixed RGB is not a teacher source."""
    if readonly.shape!=ideal_rgb.shape or baseline.shape!=readonly.shape or eligible_weight.shape!=readonly.shape[1:] or valid.shape!=eligible_weight.shape:raise ValueError('conditional target shape mismatch')
    target=readonly.astype(np.float64)+eligible_weight[None].astype(np.float64)*ideal_rgb
    return np.where(valid[None],target,baseline).astype(np.float32)

def sample_world_ideal(maps,surface,supported,source_valid,cases,config,evaluate_world,iter_pixel_chunks,*,density=2.,point_budget=131072):
    center=maps['center_uv'];jac=maps['pixel_uv_jacobian'];n=len(center)
    rgb=np.zeros((len(cases),n,3),np.float64);paint=np.zeros((len(cases),n),np.float64);mass=np.zeros(n,np.float64);geom_mass=np.zeros(n,np.float64)
    total_points=0;max_q=0
    for selected,uv,w,meta in iter_pixel_chunks(center,jac,density=density,point_budget=point_budget):
        xyz,gv,_=triangle_world(uv,surface,supported)
        sw=gv*bilinear_valid(uv,source_valid)*w[None];m=sw.sum(1)
        mass[selected]=m;geom_mass[selected]=(gv*w[None]).sum(1)
        for i,case in enumerate(cases):
            color,p=evaluate_world(xyz,case,config)
            rgb[i,selected]=np.divide((color*sw[...,None]).sum(1),m[:,None],out=np.zeros((len(selected),3)),where=m[:,None]>1e-12)
            paint[i,selected]=np.divide((p*sw).sum(1),m,out=np.zeros(len(selected)),where=m>1e-12)
        total_points+=int(meta['points']);max_q=max(max_q,int(meta['nx']*meta['ny']))
    return rgb,paint,mass,geom_mass,dict(points=total_points,max_points_per_pixel=max_q,density=density)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--highres',type=Path,default=HIGH);p.add_argument('--ledger',type=Path,default=Q/'permission/final_r1/teacher_evidence.npz')
    p.add_argument('--function',type=Path,default=IDEAL/'ideal_code/ideal_world_teacher.py');p.add_argument('--config',type=Path,default=IDEAL/'ideal_targets_r2/field_config.json')
    p.add_argument('--pixel-helper',type=Path,default=Q/'pixel_render/code/pixel_quadrature_stream.py');p.add_argument('--output',type=Path,default=NEW/'qa/ideal_world_control_r1')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('preserve previous ideal preparation; fresh output required')
    started=time.perf_counter();ideal=import_file('independent_ideal_world_function',args.function);pixel=import_file('independent_pixel_box_stream',args.pixel_helper)
    cfg=json.loads(args.config.read_text('utf-8'))
    cases=[dict(name='asphalt',asphalt='lowfreq',marking=None)]+[c for c in cfg['cases'] if c['name'] in ['marking_w120','marking_w240']]
    if [c['name'] for c in cases]!=['asphalt','marking_w120','marking_w240']:raise ValueError('exact historical12/24cm cases required')
    atlas=load_keys(CANONICAL/'base_atlas.npz',['surface_world_xyz','grid_world_row']);surface=atlas['surface_world_xyz']
    support=load_keys(CANONICAL/'provenance.npz',['surface_supported'])['surface_supported'][atlas['grid_world_row']]
    valid=load_keys(args.ledger,['common_material_valid'])['common_material_valid']
    if valid.shape!=surface.shape[:2]:raise ValueError('canonical source-valid mismatch')
    inputs=[args.function,args.config,args.pixel_helper,args.ledger,CANONICAL/'base_atlas.npz',CANONICAL/'provenance.npz']
    source_hash={str(x):sha(x) for x in inputs};args.output.mkdir(parents=True);rows=[]
    for raw in [86,110]:
        path=args.highres/f'maps/{raw:03d}.npz'
        maps=load_keys(path,['pixel_ids','center_uv','pixel_uv_jacobian','surface_world_xyz','surface_hit','image_hw'])
        ids=maps['pixel_ids'];shape=tuple(maps['image_hw'].astype(int));n=np.prod(shape)
        center_xyz,center_ok,J=triangle_world(maps['center_uv'],surface,support)
        error=float(np.abs(center_xyz-maps['surface_world_xyz']).max())
        if error>1e-7 or not center_ok.all():raise ValueError('canonical UV->actual G_s source identity failed')
        rgb,paint,mass,gm,cost=sample_world_ideal(maps,surface,support,valid,cases,cfg,ideal.evaluate_world,pixel.iter_pixel_chunks)
        available=np.zeros(n,bool);available[ids]=mass>1e-12;available=available.reshape(shape)
        dense_rgb=np.zeros((len(cases),3,n),np.float32);dense_paint=np.zeros((len(cases),n),np.float32)
        dense_rgb[:,:,ids]=rgb.transpose(0,2,1).astype(np.float32);dense_paint[:,ids]=paint.astype(np.float32)
        dense_rgb=dense_rgb.reshape(len(cases),3,*shape);dense_paint=dense_paint.reshape(len(cases),*shape)
        out=dict(pixel_ids=ids,ideal_rgb=dense_rgb,paint_fraction=dense_paint,source_mass_sparse=mass,geometry_mass_sparse=gm,
                 ideal_available=available,canonical_hit=maps['surface_hit'],image_hw=np.asarray(shape),cases=np.asarray([c['name'] for c in cases]),
                 source_function=np.asarray(str(args.function)),source_config=np.asarray(str(args.config)))
        Jxy=np.einsum('nij,njk->nik',J[:,:2],maps['pixel_uv_jacobian']).swapaxes(-1,-2)
        for case in cases[1:]:
            qual=ideal.sampling_qualification(center_xyz,Jxy,case,cfg)
            for key in ['width_px','on_px','off_px','resolvable','nyquist_ok','line_corridor']:
                out[case['name']+'_'+key+'_sparse']=qual[key]
        basis_rows={}
        for mode in ['road','joint']:
            basis_path=args.highres/f'native_r1/{raw:03d}/{mode}_basis.npz'
            b=load_keys(basis_path,['readonly_rgb','eligible_weight','baseline_rgb','alpha','K'])
            if mode=='joint' and np.any(b['eligible_weight'][b['K']]!=0):
                raise ValueError('cached eligible joint weight overlaps K; report scope conflict, never shrink it silently')
            targets=np.stack([compose_conditional_target(b['readonly_rgb'],b['eligible_weight'],c,available,b['baseline_rgb']) for c in dense_rgb])
            out[mode+'_target_P']=targets;out[mode+'_target_paint_increment']=targets[1:]-targets[0:1]
            out[mode+'_eligible_weight']=b['eligible_weight'];out[mode+'_fixed_alpha']=b['alpha'];out[mode+'_K']=b['K']
            basis_rows[mode]=dict(native_basis=str(basis_path),We_zero=int((b['eligible_weight']==0).sum()),We_ge95=int((b['eligible_weight']>=.95).sum()),
                                  alpha_K_source='cached actual same-sort native basis',K_target_baseline_difference=float(np.abs(targets-b['baseline_rgb'][None])[:,:,b['K']].max()) if b['K'].any() else 0.)
            source_hash[str(basis_path)]=sha(basis_path)
        dest=args.output/f'{raw:03d}.npz';np.savez_compressed(dest,**out)
        source_hash[str(path)]=sha(path)
        r=dict(raw=raw,pixels=int(n),canonical_hits=len(ids),ideal_available=int(available.sum()),source_mass0_on_canonical=int((mass<=1e-12).sum()),canonical_center_reconstruction_max_m=error,
               quadrature=cost,basis=basis_rows,output=str(dest),output_sha256=sha(dest));rows.append(r)
        print(json.dumps(dict(raw=raw,prepared=True,points=cost['points'],available=r['ideal_available'])),flush=True)
    receipt=dict(schema='independent_world_ideal_conditional_capacity.v1',complete=True,frames=rows,cases=[c['name'] for c in cases],source_SHA256=source_hash,code_SHA256=sha(__file__),seconds=time.perf_counter()-started,
        target_formula='P*=P_readonly+W_eligible*A_ideal_pixelbox; each road/joint uses its own native fixed weights. Fallback uses cached native baseline and is not ideal capability evidence.',
        independent_teacher='historical analytic world asphalt+known12/24cm curved dashed paint; evaluated continuously at actual canonical-surface UV samples, never GS-rendered teacher or2cm RGB-node interpolation',
        antialias='same uncapped density2 anisotropic pixel-box UV differential; sample XYZ follows actual piecewise canonical triangle topology. Subpixel ray map is the registered local differential, not an exact retrace at each subpixel.',
        nontrivial_gate='Full registered denominator remains reported. Nontrivial line detail must additionally show fixed We>=.95, source available, >=3native-pixel width/on/off qualification; We0 target identity is not expressive success.',
        contrast='Paint minus common asphalt target cancels readonly exactly up to stored float rounding. Do not compare uncorrected full-scene contrast as if readonly were ideal paint.',
        ground_truth_role='synthetic controlled world function, not hidden-road ground truth or a blind test',
        GPU_executed=False,native_calls=0,optimizer_updates=0,production_asset_modified=False)
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf-8')
    print(json.dumps(dict(completed=True,output=str(args.output),seconds=receipt['seconds'])))

if __name__=='__main__':main()
