"""One fixed 6-world-basis /18-RGB-coefficient seam material control.

GPU only measures exact alphaT basis columns in the frozen n2 scene. CPU solves
one convex constrained least squares problem per channel. No alpha, geometry,
camera, old-road/native/static/env or deleted-hidden RGB is fitted.
"""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import argparse,hashlib,json,shutil,sys,time
import numpy as np

RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_road_seam_contact_20261107_r1')
PARENT=RUN.parent/'scene008_road_r0_v44_20261107_r1'
DEST=RUN/'blue/boundary_material_v1'
ASSET=RUN/'engineer/inputs/new_road_canonical_candidate.npz'
MANIFEST=PARENT/'candidates/baked_cellfit_strip1000_candidate_v1/scene_manifest.json'
MASKS=PARENT/'surface/material_cellfit_strip_v1/supervision_masks.npz'
HELD=(89,95,101,105,113,121,125,130,137)
BUILD=tuple(x for x in range(89,138) if x not in HELD)
SMOKE=(90,91)
RIDGE=.01
PROTECT_WEIGHT=1.


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def load(p):
    with np.load(p,allow_pickle=False) as d:return {k:d[k].copy() for k in d.files}


def prepare():
    DEST.mkdir(parents=True,exist_ok=False)
    a=load(ASSET);xyz=a['local_xyz'].astype(float)
    low=np.maximum(0,a['original_source_proxy_srgb'].astype(float)-.15)
    high=np.minimum(1,a['original_source_proxy_srgb'].astype(float)+.15)
    base=a['rgb'].astype(float)
    assert base.shape==(10098,3) and np.all(base>=low-1e-7) and np.all(base<=high+1e-7)
    # Three geometry boundaries are frozen before this control; not chosen from residuals.
    # Near/far tangent is y normalized over entire parent strip; lateral tangent is x.
    tangent_y=np.clip((xyz[:,1]-(-.32))/3.,-1,1)
    tangent_x=np.clip((xyz[:,0]-25.8)/6.,-1,1)
    basis=np.stack([v for axis,edge,t in [(0,19.8,tangent_y),(0,31.8,tangent_y),(1,2.71,tangent_x)]
                    for v in [np.maximum(0,1-np.abs(xyz[:,axis]-edge)/.4),np.maximum(0,1-np.abs(xyz[:,axis]-edge)/.4)*t]],1)
    assert basis.shape==(10098,6) and np.isfinite(basis).all()
    np.savez_compressed(DEST/'prepared.npz',basis=basis.astype(np.float32),base_rgb=base.astype(np.float32),
                        source_proxy=a['original_source_proxy_srgb'],interval_low=low,interval_high=high,
                        local_xyz=a['local_xyz'])
    shutil.copy2(MASKS,DEST/'supervision_masks.npz')
    snap=DEST/'source_snapshot';snap.mkdir()
    for p in [Path(__file__),RUN/'engineer/refined_road_runtime.py',RUN/'engineer/world_surface_sampling.py']:
        shutil.copy2(p,snap/p.name)
    sources={str(p):sha(p) for p in [ASSET,MANIFEST,MASKS,DEST/'prepared.npz',Path(__file__),
                                  RUN/'engineer/refined_road_runtime.py',RUN/'engineer/world_surface_sampling.py']}
    c=dict(schema='BLUE_GEOMETRY_BOUNDARY_MATERIAL_6BASIS18RGB_V1',status='PREPARED_NOT_ADOPTED',
        parent='Frozen V44 + cellfit1000; fixed world n2 resampling, no RelativeMip/AA',
        BUILD40=list(BUILD),HELD9=list(HELD),smoke_raws=list(SMOKE),states=['keep','delete'],
        parameter_count=18,basis_count=6,parent_rows=10098,child_rows=40392,
        basis={'near_x_m':19.8,'far_x_m':31.8,'lateral_y_m':2.71,'compact_distance_radius_m':.4,
               'formula':'max(0,1-|axis-edge|/.4) * [1,normalized_tangent]',
               'near_far_tangent':'clip((local_y+.32)/3,-1,1)',
               'lateral_tangent':'clip((local_x-25.8)/6,-1,1)',
               'scope':'Entire fixed world boundary for all views/states, never raw128/ROI-specific',
               'child_policy':'Each frozen n2 child inherits its parent canonical-row basis and colour'},
        supervision='Exact existing original BUILD40 true-road masks: exclude visible/reviewed A6/A37, both box_candidate masks and frozen other_vehicle_guard; independent of output alpha/RGB. Original RGB only; no road_edit_rgb/depth or hidden delete labels.',
        supervision_masks_sha256=sha(DEST/'supervision_masks.npz'),
        protection='Every pixel outside the fixed data mask targets same fixed n2 parent rendered RGB (zero delta); equal per-view mean data/protection weights. All original fields/geometry/alpha remain frozen.',
        objective='Mean over80 fixed frame/state views of [mean_data ||Mcoef-(originalRGB-parentRGB)||² + 1*mean_outside ||Mcoef||²] + .01*mean_parentcells ||Bcoef||², independently per RGB channel.',
        protect_weight=PROTECT_WEIGHT,ridge=RIDGE,
        constraints='For every canonical parent row and RGB: max(0,original_source_proxy-.15)<=baseRGB+Bcoef<=min(1,original_source_proxy+.15). Zero initial coef feasible; linear constraints; no posthoc clipping.',
        source_delta_bound=.15,zero_initial_feasible=True,
        source_proxy_is_GT=False,source_proxy_seen_all199=True,
        held_usage='HELD9 not used for fitting, lambda/budget/method selection. Previous scene has already been opened; same-scene development, no independent confirmation claim.',
        GPU_measurement='24feature ABI first18 original group tags/RGB retained; newroad feature18:24 stores 6basis; extra16:22 is alphaT-weighted basis M. Colours remain true; one shared sort.',
        lifecycle='Smoke raw90 two states -> save normal equations -> new process restore and add raw91 two states -> constrained fit entry. Formal fresh80F separate. Final two-raw four-state validation checks predicted RGB=base+Mcoef against actual rendering.',
        formal_forwards=80,smoke_forwards=4,validation_forwards=4,training_backwards=0,
        scope_limits=['Material compensation can absorb static/newroad blend; does not solve or certify depth ordering',
                      'No hidden road texture GT, no shadow/marking GT, no learned prior or generalization claim',
                      'Blue and red acceptance separate; final whole-scene adoption remains parent decision'],
        immutable_sources=sources,source_snapshots={p.name:sha(p) for p in snap.iterdir()},
        basis_active_rows=int((abs(basis).sum(1)>0).sum()),basis_column_active_rows=(abs(basis)>0).sum(0).astype(int).tolist(),
        source_proxy_initial_maxabs=float(abs(base-a['original_source_proxy_srgb']).max()))
    write(DEST/'contract.json',c)
    print(json.dumps({k:c[k] for k in ['basis_active_rows','basis_column_active_rows','source_proxy_initial_maxabs']},ensure_ascii=False),flush=True)


def frozen():
    c=json.loads((DEST/'contract.json').read_text())
    for p,s in c['immutable_sources'].items():assert sha(p)==s,p
    assert sha(DEST/'supervision_masks.npz')==c['supervision_masks_sha256']
    return c,load(DEST/'prepared.npz'),load(DEST/'supervision_masks.npz')


def setup():
    import torch
    c,p,m=frozen();torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.20)
    sys.path.insert(0,str(DEST/'source_snapshot'))
    from refined_road_runtime import RefinedRoadRuntime
    model=RefinedRoadRuntime(MANIFEST,factor=2,verify_dependencies=False)
    assert model.opacity_fn is None and np.array_equal(model.cpu_parent['local_xyz'],p['local_xyz'])
    return torch,c,p,m,model


def render_basis(torch,model,p,raw,state,colours=None):
    scene=model.scene(raw,delete=(state=='delete'))
    rows=dict(scene['rows']);feat=rows['features'].clone();select=feat[:,18]>0
    ids=model.cpu_child['parent_indices']
    assert int(select.sum())==len(ids)==40392
    feat[:,18:24]=0
    feat[select,18:24]=torch.as_tensor(p['basis'][ids],device='cuda')
    if colours is not None:
        cols=rows['colors_precomp'].clone();cols[select]=torch.as_tensor(colours[ids],device='cuda')
        rows['colors_precomp']=cols;feat[select,12:15]=cols[select]
    rows['features']=feat;scene=dict(scene,rows=rows)
    out=model.rt.render(raw,scene)
    return out


def prepass(name,raws,resume):
    torch,c,p,m,model=setup();folder=DEST/name
    statepath=folder/'normal_equations.npz';recordpath=folder/'receipt.json'
    if resume:
        previous=json.loads(recordpath.read_text());assert sha(statepath)==previous['normal_equations_sha256']
        z=load(statepath);hd=z['H_data'];hp=z['H_protect'];g=z['g'];records=previous['records'];count=int(z['views'])
        restored_sha=sha(statepath)
    else:
        folder.mkdir(exist_ok=False);hd=np.zeros((6,6));hp=hd.copy();g=np.zeros((6,3));count=0;records=[];restored_sha=None
    assert not set(raws)-set(BUILD)
    assert not set(raws)&{r['raw'] for r in records}
    start=time.time();before=model.rt.render_calls
    with torch.no_grad():
        for raw in raws:
            original=np.asarray(model.rt.raw(raw)['obs']['rgb'],np.float64).transpose(2,0,1)
            if original.max()>1:original/=255
            mask=m[f'raw{raw:03d}'].astype(bool);outside=~mask
            obs=model.rt.raw(raw)['obs']
            assert not (mask&obs['visible'].any(0)).any()
            assert not (mask&obs['box_candidate'].any(0)).any()
            for state in ('keep','delete'):
                out=render_basis(torch,model,p,raw,state)
                rgb=out['rgb'].cpu().numpy().astype(float)
                M=out['extra'][16:22].cpu().numpy().astype(float).transpose(1,2,0)
                target=(original-rgb).transpose(1,2,0)
                xd=M[mask];yp=target[mask];xp=M[outside]
                hd+=xd.T@xd/max(1,len(xd));hp+=xp.T@xp/max(1,len(xp));g+=xd.T@yp/max(1,len(xd));count+=1
                rec=dict(raw=int(raw),state=state,data_pixels=int(mask.sum()),protect_pixels=int(outside.sum()),
                    initial_data_RGB_MSE=float((yp**2).mean()),column_L2_data=(xd**2).sum(0).tolist(),
                    column_L2_protect=(xp**2).sum(0).tolist(),group_RGB_conservation=float(out['color_reconstruction_maxabs']))
                records.append(rec)
                if raw in SMOKE:
                    np.savez_compressed(folder/f'probe_{raw:03d}_{state}.npz',base_rgb=rgb.astype(np.float32),M=M.astype(np.float32),original_rgb=original.astype(np.float32),data_mask=mask)
                del out,M,rgb
            model.rt._data.clear();torch.cuda.empty_cache()
            np.savez_compressed(statepath,H_data=hd,H_protect=hp,g=g,views=np.array(count))
            receipt=dict(status='BASIS_PREFORWARD_COMPLETE',name=name,views=count,records=records,
                normal_equations_sha256=sha(statepath),resumed_from_sha256=restored_sha,
                appended_forwards=model.rt.render_calls-before,seconds_this_process=time.time()-start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                contract_sha256=sha(DEST/'contract.json'),runtime=model.receipt(),
                train_backwards=0,original_parameter_updates=0,basis_only_extra_features=True)
            write(recordpath,receipt)
            print(json.dumps(dict(name=name,raw=raw,views=count,seconds=time.time()-start)),flush=True)
    frozen()


def solve(name):
    from scipy.optimize import minimize,LinearConstraint
    c,p,m=frozen();folder=DEST/name;receipt=json.loads((folder/'receipt.json').read_text())
    assert sha(folder/'normal_equations.npz')==receipt['normal_equations_sha256']
    z=load(folder/'normal_equations.npz');n=int(z['views'])
    if name=='formal':assert n==80 and {r['raw'] for r in receipt['records']}==set(BUILD)
    B=p['basis'].astype(float);base=p['base_rgb'].astype(float)
    H=(z['H_data']+PROTECT_WEIGHT*z['H_protect'])/n+RIDGE*(B.T@B)/len(B)
    H=(H+H.T)/2;g=z['g']/n;active=abs(B).sum(1)>0
    coef=np.zeros((6,3));solver=[]
    for channel in range(3):
        lo=(p['interval_low']-base)[:,channel];hi=(p['interval_high']-base)[:,channel]
        constraints=LinearConstraint(B[active],lo[active],hi[active])
        result=minimize(lambda x:.5*x@H@x-g[:,channel]@x,np.zeros(6),jac=lambda x:H@x-g[:,channel],
                        method='SLSQP',constraints=[constraints],options={'ftol':1e-12,'maxiter':500,'disp':False})
        if not result.success:raise RuntimeError(str(result))
        coef[:,channel]=result.x
        solver.append(dict(channel=channel,success=bool(result.success),iterations=int(result.nit),message=str(result.message),objective=float(result.fun),gradient_norm=float(np.linalg.norm(result.jac))))
    colours=base+B@coef
    violation=max(float(np.max(p['interval_low']-colours)),float(np.max(colours-p['interval_high'])),0.)
    assert violation<=1e-8
    assert np.array_equal(colours[~active].astype(np.float32),p['base_rgb'][~active])
    np.savez_compressed(folder/'material_override.npz',colours_srgb=colours.astype(np.float32),coefficients_rgb=coef,
                        basis=p['basis'],parent_asset_sha256=np.array(sha(ASSET)),source_proxy=p['source_proxy'])
    write(folder/'fit_receipt.json',dict(status='CONSTRAINED_WORLD_MATERIAL_SOLVED_NOT_ADOPTED',name=name,
        views=n,normal_equations_sha256=sha(folder/'normal_equations.npz'),contract_sha256=sha(DEST/'contract.json'),
        material_override_sha256=sha(folder/'material_override.npz'),solver=solver,coefficients_rgb=coef.tolist(),
        eigenvalues_H=np.linalg.eigvalsh(H).tolist(),constraint_max_violation_float64=violation,
        colour_delta_maxabs=float(abs(colours-base).max()),source_proxy_final_maxabs=float(abs(colours-p['source_proxy']).max()),
        active_rows=int(active.sum()),outside_basis_exactly_frozen=True,
        formal_method_changed=False,held9_used_in_fit=False,same_scene_development=True,RGB_linear_validation_pending=True))
    print(json.dumps(dict(name=name,views=n,coef=coef.tolist(),source_proxy_final_maxabs=float(abs(colours-p['source_proxy']).max()))),flush=True)


def validate(name):
    torch,c,p,m,model=setup();folder=DEST/name;fit=load(folder/'material_override.npz');records=[];start=time.time()
    with torch.no_grad():
        for raw in SMOKE:
            for state in ('keep','delete'):
                probe=load(folder/f'probe_{raw:03d}_{state}.npz')
                out=render_basis(torch,model,p,raw,state,colours=fit['colours_srgb'])
                rendered=out['rgb'].cpu().numpy();pred=probe['base_rgb']+np.einsum('hwk,kc->chw',probe['M'],fit['coefficients_rgb'])
                diff=abs(rendered-pred)
                assert diff.max()<1e-5,(raw,state,diff.max())
                records.append(dict(raw=raw,state=state,linear_prediction_render_maxabs=float(diff.max()),meanabs=float(diff.mean()),
                                    RGB_conservation=float(out['color_reconstruction_maxabs'])))
            model.rt._data.clear();torch.cuda.empty_cache()
    write(folder/'linear_validation.json',dict(status='PASS_REAL_RENDER_LINEAR_SUPERPOSITION',records=records,
        forwards=4,seconds=time.time()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        original_parameter_updates=0,geometry_alpha_updates=0,material_override_sha256=sha(folder/'material_override.npz'),runtime=model.receipt()))
    print(json.dumps(records),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','prepass','solve','validate'])
    ap.add_argument('--name',default='smoke');ap.add_argument('--raws',nargs='+',type=int);ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='prepass':prepass(args.name,args.raws or list(BUILD),args.resume)
    elif args.stage=='solve':solve(args.name)
    else:validate(args.name)


if __name__=='__main__':main()
