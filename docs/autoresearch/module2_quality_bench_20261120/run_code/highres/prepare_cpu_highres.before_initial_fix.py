"""True native-pixel mesh/footprint maps and frozen fields. CPU only, zero updates."""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,sys,hashlib,time,gc,traceback,shutil
import numpy as np
from PIL import Image

B=Path('/root/autodl-tmp/drivingeditor_repro/campaign');Q=B/'road_teacher_detail_cpu_20261119_r1';G=Q/'gpu_20261119'
DEFAULT=G/'quality_bench_20261120_r1/highres';R=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
ASSET=B/'road_activation_20261118_r1/B32_beam95_native_r1/asset_guarded'
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p,keys=None):
    with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in(z.files if keys is None else keys)}
def rawrgb(p):return np.asarray(Image.open(p).convert('RGB')).transpose(2,0,1).astype(np.float32)/255
def nnmask(a):return np.repeat(np.repeat(a,4,axis=0),4,axis=1)

def prepare(out):
    import torch
    torch.set_num_threads(1);began=time.perf_counter();out=Path(out)
    if(out/'cpu_prepared.json').exists():raise FileExistsError('preserve completed preparation')
    for d in ['targets','maps','pixel_reads','rendered']:(out/d).mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[str(R/'scripts/worldsim_v82'),str(Q/'code'),str(Q/'pixel_render/code')]
    from drivingeditor_dynamic_edit.road_surface_projection import project_surface_mesh
    from fixed_material_read_operator import bilinear
    from world_texture_runtime import WorldTextureRuntime
    from pixel_quadrature_stream import iter_pixel_chunks
    manifest_path=ASSET/'persistent/scene.json';manifest=read(manifest_path);frame={r['key']:r for r in manifest['frames']}
    old_manifest=read(B/'road_direct_teacher_20261116_r1/supervision_r3/manifest.json');oldrows={r['raw']:r for r in old_manifest['rows']}
    ident=B/'road_capacity_cpu_20261117_r1/supervision_source_identity_r1'
    mesh_asset=B/'road_asset_20261116_r1/phase2/formal_r1/asset'
    atlas_data=load(mesh_asset/'base_atlas.npz',['surface_world_xyz','grid_world_row']);atlas=SimpleNamespace(**atlas_data)
    support=load(mesh_asset/'provenance.npz',['surface_supported'])['surface_supported']
    road=load(ASSET/'road.npz',['minimum_model_depth','meter_to_model','model_from_world']);grid=atlas.grid_world_row
    xyz_world=np.empty((grid.size,3),np.float64);xyz_world[grid.ravel()]=atlas.surface_world_xyz.reshape(-1,3)
    yy,xx=np.indices(grid.shape);uv_world=np.empty((grid.size,2),np.float64);uv_world[grid.ravel()]=np.stack([xx.ravel(),yy.ravel()],1)
    field_paths=dict(initial=G/'formal_r1/pixel_initial/field.pt',low_A200=G/'formal_r1/pixel_common_A200/field.pt',
        M01=G/'formal_r1/M01/C800/field.pt',M11=G/'formal_r1/M11/C800/field.pt')
    fields={};sources={str(manifest_path):sha(manifest_path),str(mesh_asset/'base_atlas.npz'):sha(mesh_asset/'base_atlas.npz'),str(mesh_asset/'provenance.npz'):sha(mesh_asset/'provenance.npz')}
    base=valid=None
    for name,p in field_paths.items():
        model,meta=WorldTextureRuntime.load(p,device='cpu')
        with torch.no_grad():L,H=model.fields()
        if base is None:base=model.base.numpy().copy();valid=model.valid.numpy()[...,0].copy()
        else:assert np.array_equal(base,model.base.numpy())and np.array_equal(valid,model.valid.numpy()[...,0])
        fields[name]=(L.numpy().copy(),H.numpy().copy(),model.H_gate.numpy()[...,0].copy());sources[str(p)]=sha(p);del model
    fields['M11_noH']=(fields['M11'][0],np.zeros_like(fields['M11'][1]),fields['M11'][2])
    rows=[]
    for raw in [110,86]:
        started=time.perf_counter();mapfile=out/f'maps/{raw:03d}.npz';targetfile=out/f'targets/{raw:03d}.npz';readsfile=out/f'pixel_reads/{raw:03d}.npz'
        source=oldrows[raw]['source_files'];roles=load(source['roles']['path']);guard=load(source['guard']['path']);route=load(ident/f'{raw:03d}_native_routing_evidence.npz')
        obs=rawrgb(source['observation']['path']);teacher=rawrgb(source['teacher_C']['path']);assert obs.shape==teacher.shape==(3,1280,1920)
        for key in ['observation','teacher_C','roles','guard','road_observed']:
            p=Path(source[key]['path']);sources[str(p)]=sha(p);assert sources[str(p)]==source[key]['sha256']
        road_seen=np.asarray(Image.open(source['road_observed']['path']))>0
        annotations=Image.open(frame[raw]['instance_mask']);annotation_shape=list(annotations.size)
        labels=np.asarray(annotations.resize((1920,1280),Image.Resampling.NEAREST));Kv=(labels>0)&(labels!=10)
        Kt=roles['K']|guard['K'];possible=roles['shadow_possible']|roles['shadow_uncertain']|roles['shadow_boundary_guard']
        An=road_seen&roles['C_mat']&roles['visible_support']&~guard['H_B']&~Kt
        A=An&~possible;C_requested=roles['physical_unknown']&guard['W_guard']&~Kt
        C=C_requested&~route['other_shadow_prediction'];S=route['target_shadow_interior']&guard['W_guard']&~route['other_shadow_prediction']&~Kt
        assert not any((a&b).any()for a,b in [(A,C),(A,S),(C,S)])
        old=load(G/f'evaluation/frozen_r2/{raw:03d}.npz',['A','C','S'])
        if not mapfile.exists():
            low=load(Q/f'pixel_render/maps/{raw:03d}.npz',['camera_K','world_to_camera']);K=low['camera_K'].copy();K[:2]*=4;M=low['world_to_camera']
            mesh=project_surface_mesh(atlas,support,K,M,(1280,1920),road['minimum_model_depth']/float(road['meter_to_model']))
            hit=mesh['valid'];pixels=np.flatnonzero(hit);ids=mesh['triangle_world_rows'][hit];weights=mesh['triangle_weights'][hit];X=mesh['world_xyz'][hit]
            uv=(uv_world[ids]*weights[...,None]).sum(1);verts=xyz_world[ids]
            rot=M[:3,:3];origin=-rot.T@M[:3,3];py,px=np.unravel_index(pixels,hit.shape)
            direction=np.c_[((np.stack([px,py],1)+.5)-K[:2,2])/np.diag(K)[:2],np.ones(len(pixels))]@rot
            normal=np.cross(verts[:,1]-verts[:,0],verts[:,2]-verts[:,0]);den=(normal*direction).sum(1);t=(normal*(verts[:,0]-origin)).sum(1)/den
            Jxy=np.empty((len(pixels),2,2))
            for axis in [0,1]:
                dr=rot.T[:,axis]/K[axis,axis];dt=-t*(normal@dr)/den;Jxy[:,:,axis]=(t[:,None]*dr+direction*dt[:,None])[:,:2]
            Ex=np.stack([verts[:,1,:2]-verts[:,0,:2],verts[:,2,:2]-verts[:,0,:2]],-1)
            Eu=np.stack([uv_world[ids[:,1]]-uv_world[ids[:,0]],uv_world[ids[:,2]]-uv_world[ids[:,0]]],-1)
            Juv=np.einsum('nij,njk->nik',Eu@np.linalg.inv(Ex),Jxy)
            np.savez_compressed(mapfile,pixel_ids=pixels,center_uv=uv,pixel_uv_jacobian=Juv,surface_world_xyz=X,triangle_world_rows=ids,triangle_weights=weights,
                surface_hit=hit,camera_K=K,world_to_camera=M,pixel_center_offset=np.asarray(.5),image_hw=np.asarray([1280,1920]),
                reprojection_error_px_max=np.asarray(np.nanmax(mesh['reprojection_error_px'])))
            del mesh,verts,ids,weights,direction,normal,Jxy,Ex,Eu;gc.collect()
        maps=load(mapfile);pixels=maps['pixel_ids'];hit=maps['surface_hit']
        if not readsfile.exists():
            colors={name:np.zeros((len(pixels),3),np.float32)for name in fields};mass_all=np.zeros(len(pixels),np.float64);points=0
            # Complete pixel rows are never split: source normalization is per actual pixel footprint.
            for positions,uv,w,info in iter_pixel_chunks(maps['center_uv'],maps['pixel_uv_jacobian'],density=2.,point_budget=131072):
                V=bilinear(valid,uv)[...,0];mass=(V*w[None]).sum(1);base_num=(bilinear(base*valid[...,None],uv)*w[None,:,None]).sum(1)
                mass_all[positions]=mass;points+=info['points']
                for name,(L,H,gate)in fields.items():
                    val=bilinear(L,uv/6)+bilinear(gate,uv)*bilinear(H,uv)
                    num=base_num+(V[...,None]*val*w[None,:,None]).sum(1)
                    colors[name][positions]=np.clip(num/np.maximum(mass[:,None],1e-12),0,1).astype(np.float32)
            np.savez_compressed(readsfile,pixel_ids=pixels,source_mass=mass_all,quadrature_points=np.asarray(points),**colors)
        reads=load(readsfile);mass=np.zeros(1280*1920,np.float64);mass[pixels]=reads['source_mass'];mass=mass.reshape(1280,1920)
        T=dict(T_A=obs,T_C=teacher,T_S=teacher,A_pre_alpha=A,C_pre_alpha=C,S_pre_alpha=S,K=Kv,K_vehicle=Kv,K_teacher_guard=Kt,
            surface_hit=hit,source_mass=mass,source_material_valid=mass>1e-12,geometry_missing=(A|C|S)&~hit,
            expected_legal_road=A|C|S,C_requested_native=C_requested,C_other_shadow_excluded=C_requested&route['other_shadow_prediction'],
            S_teacher_missing=route['target_shadow_interior']&~guard['W_guard'],S_boundary_unknown=roles['shadow_uncertain']|roles['shadow_boundary_guard'],
            legacy480_A_texture_domain=nnmask(old['A']),legacy480_C_texture_domain=nnmask(old['C']),legacy480_S_texture_domain=nnmask(old['S']))
        if not targetfile.exists():np.savez_compressed(targetfile,**T)
        counts={r:int(T[r+'_pre_alpha'].sum())for r in ['A','C','S']}
        rows.append(dict(raw=raw,pre_alpha=counts,K_vehicle=int(Kv.sum()),K_teacher_guard=int(Kt.sum()),surface_hit=int(hit.sum()),
            surface_hit_by_role={r:int((T[r+'_pre_alpha']&hit).sum())for r in ['A','C','S']},
            source_valid_by_role={r:int((T[r+'_pre_alpha']&(mass>1e-12)).sum())for r in ['A','C','S']},
            true_native_geometry_reprojection_max=float(maps['reprojection_error_px_max']),quadrature_points=int(reads['quadrature_points']),
            native_K_annotation_source_size=annotation_shape,seconds=time.perf_counter()-started))
        print('CPU_FRAME_PREPARED',raw,rows[-1],flush=True)
        del roles,guard,route,obs,teacher,reads,T,maps,mass;gc.collect()
    write(out/'cpu_prepared.json',dict(completed=True,rows=rows,seconds=time.perf_counter()-began,sources_SHA256=sources,fields={k:str(v)for k,v in field_paths.items()},
        case_roles=dict(M01='formal',M11='formal',M10='formal actual GS average',initial='stage diagnostic',low_A200='stage diagnostic',M11_noH='fixed finalL ablation'),
        primary_masks='original native legal target masks before surface/source/alpha tests; actual available masks reported separately',
        G_s='fresh canonical triangle raster at every1920 pixel centre; old480 UV is never resized; same frozen world mesh',
        footprint='fresh per-triangle local pixel differential; density2 uncapped anisotropic midpoint box, source normalized; no Gaussian Sigma in pixel filter',
        native_calls=0,optimizer_updates=0,GPU_operations=0))

def compose(out):
    out=Path(out)
    if not(out/'cpu_prepared.json').exists()or not(out/'native_completed.json').exists():raise RuntimeError('native and CPU input both required')
    rows=[]
    for raw in [110,86]:
        reads=load(out/f'pixel_reads/{raw:03d}.npz');ids=reads['pixel_ids'];valid=reads['source_mass']>1e-12
        basis_road=load(out/f'native_r1/{raw:03d}/road_basis.npz');basis_joint=load(out/f'native_r1/{raw:03d}/joint_basis.npz')
        fallback=np.ones(1280*1920,bool);fallback[ids]=~valid;fallback=fallback.reshape(1280,1920)
        for name in ['initial','low_A200','M01','M11','M11_noH']:
            folder=out/'rendered'/name;folder.mkdir(exist_ok=True);dest=folder/f'{raw:03d}.npz'
            if dest.exists():continue
            color=np.zeros((3,1280*1920),np.float32);color[:,ids]=reads[name].T;color=color.reshape(3,1280,1920)
            r=basis_road['baseline_rgb']+np.where(~fallback[None],basis_road['eligible_weight'][None]*color-basis_road['eligible_rgb'],0)
            j=basis_joint['baseline_rgb']+np.where(~fallback[None],basis_joint['eligible_weight'][None]*color-basis_joint['eligible_rgb'],0)
            jr=basis_joint['whole_road_rgb']+np.where(~fallback[None],basis_joint['eligible_weight'][None]*color-basis_joint['eligible_rgb'],0)
            np.savez_compressed(dest,road_rgb=r,joint_rgb=j,joint_road_rgb=jr,road_alpha=basis_road['alpha'],joint_alpha=basis_joint['alpha'],K=basis_joint['K'],
                road_eligible_alphaT=basis_road['eligible_weight'],joint_eligible_alphaT=basis_joint['eligible_weight'],joint_whole_road_alphaT=basis_joint['entire_road_weight'],
                fallback_mask=fallback,source_basis_joint=np.asarray(str(out/f'native_r1/{raw:03d}/joint_basis.npz')))
            rows.append(dict(raw=raw,case=name,path=str(dest),joint_K_max=float(np.abs(j-basis_joint['baseline_rgb'])[:,basis_joint['K']].max()),
                alpha_exact=True,source_attribution='static/dynamic/env are immutable sidecar arrays; joint_road_rgb includes actual joint eligible replacement'))
        # Enrich the actual-native M10 output only after preserving its original bytes.
        p=out/f'rendered/M10/{raw:03d}.npz'
        if p.exists():
            z=load(p)
            if 'joint_road_rgb'not in z:
                backup=out/f'native_r1/{raw:03d}/M10_native_original.npz';shutil.copy2(p,backup)
                z['joint_road_rgb']=z['joint_rgb']-basis_joint['static_rgb']-basis_joint['dynamic_rgb']-basis_joint['env_rgb']
                z.update(road_eligible_alphaT=basis_road['eligible_weight'],joint_eligible_alphaT=basis_joint['eligible_weight'],joint_whole_road_alphaT=basis_joint['entire_road_weight'],
                    source_basis_joint=np.asarray(str(out/f'native_r1/{raw:03d}/joint_basis.npz')))
                np.savez_compressed(p,**z)
        del reads,basis_road,basis_joint;gc.collect()
    write(out/'rendered_completed.json',dict(completed=True,cases=['M01','M11','M10','initial','low_A200','M11_noH'],frames=[110,86],rows=rows,
        image_shape=[3,1280,1920],source_attribution_sidecars='native_r1/{raw}/joint_basis.npz: static_rgb,dynamic_rgb,env_rgb,static_alphaT,dynamic_alphaT,whole_road_rgb,entire_road_weight',
        new_native_calls=0,optimizer_updates=0,GPU_operations=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--compose-only',action='store_true');a=p.parse_args()
    try:
        if not a.compose_only:prepare(a.output)
        compose(a.output)
    except Exception:
        write(a.output/('cpu_failure_'+str(int(time.time()))+'.json'),dict(traceback=traceback.format_exc(),GPU_operations=0));raise
