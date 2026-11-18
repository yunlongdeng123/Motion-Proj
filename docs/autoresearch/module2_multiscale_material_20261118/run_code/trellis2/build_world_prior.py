"""TRELLIS新mesh逐faceUV→共同A0节点；另保留任意world quad精确查询数据。"""
from pathlib import Path
import os,json,shutil
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image
from exact_mesh_uv import WorldMeshUVMap
P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1')

def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU only')
    inp=P/'trellis2/input_r1';run=P/'trellis2/inference_r1';out=P/'trellis2/world_prior_r1'
    if out.exists():raise FileExistsError('preserve old bridge')
    fixed=load(inp/'fixed_geometry.npz');generated=load(run/'generated_mesh.npz');ref=load(inp/'reference_sources.npz')
    transform=np.asarray(json.loads((inp/'transforms.json').read_text())['official_output_yup_to_world'])
    world_generated=generated['vertices']@transform[:3,:3].T+transform[:3,3]
    distance,near=cKDTree(fixed['world_vertices']).query(world_generated)
    if distance.max()>1e-5:raise ValueError('generated geometry changed')
    gen_source_ids=near[generated['faces']];lookup={tuple(sorted(t)):i for i,t in enumerate(gen_source_ids)}
    if len(lookup)!=len(fixed['faces']):raise ValueError('face identity changed/duplicated')
    matching=np.array([lookup[tuple(sorted(t))] for t in fixed['faces']],np.int32)
    uv=np.empty((len(matching),3,2))
    for fid,(source_face,gid) in enumerate(zip(fixed['faces'],matching)):
        gen_face=generated['faces'][gid];source_order=near[gen_face]
        order=np.argmax(source_order[None,:]==source_face[:,None],axis=1)
        uv[fid]=generated['uv'][gen_face[order]]
    texture=np.asarray(Image.open(run/'basecolor_rgba.png').convert('RGBA'),np.float64)/255
    mapper=WorldMeshUVMap(fixed['world_vertices'],fixed['faces'],uv,texture,fixed_to_generated_face=matching)
    node=mapper.query(fixed['world_vertices'][:,:2])
    if not node['valid'].all() or np.max(node['worldXY_reconstruction_error'])>1e-9:raise ValueError('fixed node exact inverse failed')
    A0=load(P/'cpu_initial_r1/A0.npz');grid=A0['grid_world_row'];reverse=np.empty(grid.size,np.int64);reverse[grid.ravel()]=np.arange(grid.size)
    flat=reverse[fixed['source_world_rows']];jj,ii=np.unravel_index(flat,grid.shape)
    patch=np.zeros(grid.shape,bool);patch[jj,ii]=True
    rgb=A0['rgb_srgb'].copy();rgba=node['basecolor_rgba'].astype(np.float32)
    shared=A0['valid']&patch;unknown=patch&~A0['valid']
    # 不扩共同有效材质域；generated-on-A0-unknown独立保存，不能静默扩大对照scope。
    rgb[jj,ii]=np.where(A0['valid'][jj,ii,None],rgba[:,:3],rgb[jj,ii])
    modified=np.any(rgb!=A0['rgb_srgb'],axis=2)
    if (modified&~shared).any():raise ValueError('A0 exterior changed')
    out.mkdir(parents=True)
    shutil.copy2(run/'basecolor_rgba.png',out/'basecolor_rgba.png')
    np.savez_compressed(out/'mesh_uv_source.npz',world_vertices=fixed['world_vertices'],fixed_faces=fixed['faces'],
        source_world_rows=fixed['source_world_rows'],source_world_keys=fixed['world_keys'],generated_uv_by_fixed_face=uv,
        fixed_to_generated_face=matching,generated_vertex_to_fixed=near,generated_vertices_world=world_generated,generated_faces=generated['faces'],
        reference_valid=ref['valid'],reference_source_role=ref['source_role'],reference_source_raw=ref['source_raw'],reference_source_uv_native=ref['source_uv_native'])
    np.savez_compressed(out/'node_queries.npz',source_world_rows=fixed['source_world_rows'],atlas_uv=np.c_[ii,jj],
        fixed_face_id=node['fixed_face_id'],generated_face_id=node['generated_face_id'],barycentric=node['barycentric'],generated_uv=node['generated_uv'],
        basecolor_rgba=rgba,world_xyz=node['world_xyz'],worldXY_reconstruction_error=node['worldXY_reconstruction_error'],
        reference_valid=ref['valid'],reference_source_raw=ref['source_raw'],reference_source_uv_native=ref['source_uv_native'],reference_source_role=ref['source_role'])
    combined=dict(A0);combined['rgb_srgb']=rgb
    combined['source_effective']=A0['source_effective'].copy();combined['source_effective'][shared]=4
    combined.update(trellis_patch_geometry=patch,trellis_override_valid=shared,trellis_generated_on_A0_unknown=unknown,
        trellis_material_available_node=patch,trellis_reference_unknown_node=np.zeros(grid.shape,bool))
    combined['trellis_reference_unknown_node'][jj,ii]=ref['unknown']
    np.savez_compressed(out/'A0_with_trellis_patch_nodes.npz',**combined)
    # 仅世界节点预览，不是最后GS渲染。
    image=rgba[:,:3].reshape(128,128,3).swapaxes(0,1)[::-1]
    Image.fromarray(np.rint(np.clip(image,0,1)*255).astype(np.uint8)).save(out/'patch_world_basecolor.png')
    write(out/'receipt.json',dict(schema='trellis2_exact_perface_world_prior.v1',status='PASS_CPU_GEOMETRY_UV_QUERY',
        source_inference=str(run),common_A0=str(P/'cpu_initial_r1/A0.npz'),mesh_lookup='actual fixed world XY triangle AABB index + exact barycentric + actual generated face UV; affine is not inverse truth',
        fixed_source_vertices=len(fixed['world_vertices']),fixed_faces=len(fixed['faces']),UV_split_output_vertices=len(world_generated),geometry_error_m=float(distance.max()),
        all_source_nodes_exact_inverse=True,worldXY_node_error_max_m=float(np.max(node['worldXY_reconstruction_error'])),
        patch_nodes=int(patch.sum()),common_valid_override_nodes=int(shared.sum()),generated_on_base_unknown_nodes=int(unknown.sum()),reference_unknown_nodes=int(ref['unknown'].sum()),
        A0_exterior_RGB_bytes_exact=True,common_A0_valid_unchanged=True,
        no_source_ledger_changed=True,generated_alpha_applied_to_GS_opacity=False,PBR_truth_identified=False,
        sampler_API='exact_mesh_uv.WorldMeshUVMap.load(world_prior_r1).query(world_xy[...,2]) supports arbitrary Gaussian quadrature nodes, returns exact face/bary/UV/basecolorRGBA/valid',
        source_provenance='reference UV/raw/role per fixed source vertex retained; co-located reference diagnostics are not neural causal attribution or GT',
        final_domain_classification='PENDING final shared exact-chart S_Sigma plan: patch-core/fullquadrature/boundarymix/exterior and K fallback must be recorded',
        preview_is_final_GS=False,default_asset_modified=False,GPU_operations=0,optimizer_updates=0,human_verdict=None))
    print(json.dumps(dict(output=str(out),patch_nodes=int(patch.sum()),common_valid_override_nodes=int(shared.sum()),base_unknown=int(unknown.sum()),error=float(node['worldXY_reconstruction_error'].max()))))
if __name__=='__main__':main()
