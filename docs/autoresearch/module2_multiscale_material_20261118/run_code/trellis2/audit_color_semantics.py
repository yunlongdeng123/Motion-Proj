"""原始PNG/官方GLB/真实faceUV颜色语义只读核对；不推理、不渲染。"""
from pathlib import Path
import os,sys,json
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np
from PIL import Image
import trimesh
from trimesh.visual.color import uv_to_color
from exact_mesh_uv import sample_uv_rgba
P=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_multiscale_material_20261118_r1');T=P/'trellis2'

def statistics(a):
 a=np.asarray(a,np.float64).reshape(-1,4);rgb=a[:,:3];lum=rgb@np.array([.2126,.7152,.0722])
 return dict(RGB_min=rgb.min(0).tolist(),RGB_mean=rgb.mean(0).tolist(),RGB_max=rgb.max(0).tolist(),luma_quantiles=np.quantile(lum,[0,.01,.5,.99,1]).tolist(),alpha_min_mean_max=[float(a[:,3].min()),float(a[:,3].mean()),float(a[:,3].max())])

def main():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU only')
 out=T/'color_semantics_audit_r1';out.mkdir(exist_ok=False)
 png=Image.open(T/'inference_r1/basecolor_rgba.png').convert('RGBA');raw=np.asarray(png,np.float64)/255
 with np.load(T/'inference_r1/generated_mesh.npz',allow_pickle=False) as z:g={k:z[k].copy() for k in z.files}
 with np.load(T/'world_prior_r1/node_queries.npz',allow_pickle=False) as z:node={k:z[k].copy() for k in z.files}
 glb=trimesh.load(T/'inference_r1/textured.glb',force='mesh',process=False)
 glb_image=glb.visual.material.baseColorTexture.convert('RGBA');glb_raw=np.asarray(glb_image,np.float64)/255
 rng=np.random.default_rng(101);selected=np.sort(rng.choice(len(node['generated_uv']),256,replace=False));uv=node['generated_uv'][selected]
 sampled=sample_uv_rgba(raw,uv);glb_sample=sample_uv_rgba(glb_raw,uv)
 yy=rng.integers(0,png.height,256);xx=rng.integers(0,png.width,256);centers=np.c_[(xx+.5)/png.width,1-(yy+.5)/png.height]
 ours_center=sample_uv_rgba(raw,centers);trimesh_center=uv_to_color(centers,png).astype(np.float64)/255
 nearest=uv_to_color(uv,png).astype(np.float64)/255
 our_nearest_kernel=np.c_[uv[:,0]*(png.width-1),(1-uv[:,1])*(png.height-1)]
 nx=np.round(our_nearest_kernel[:,0]).astype(int)%png.width;ny=np.round(our_nearest_kernel[:,1]).astype(int)%png.height
 expected_nearest=raw[ny,nx]
 checks=dict(GLB_vertices_same=np.array_equal(np.asarray(glb.vertices),g['vertices']),GLB_faces_same=np.array_equal(np.asarray(glb.faces),g['faces']),
  GLB_UV_same=np.array_equal(np.asarray(glb.visual.uv),g['uv']),PNG_RGBA_finite=bool(np.isfinite(raw).all()),
  query_matches_primary_PNG=bool(np.max(np.abs(sampled-node['basecolor_rgba'][selected]))<1e-6),
  normalized_texel_center_matches_trimesh_exact=bool(np.max(np.abs(ours_center-trimesh_center))<1e-12),
  trimesh_nearest_convention_exact=bool(np.array_equal(nearest,expected_nearest)),no_Torch_or_native_loaded='torch' not in sys.modules)
 receipt=dict(schema='TRELLIS2_primary_exporter_color_semantics_CPU.v1',status='PASS_PRIMARY_PNG_UV_SEMANTICS' if all(checks.values()) else 'FAIL',checks=checks,trimesh_version=trimesh.__version__,
  raw_baseColor_PNG_entire_UV_image=statistics(raw),queried_fixed_world_source_nodes=statistics(node['basecolor_rgba']),official_GLB_texture=statistics(glb_raw),
  generated_alpha_used_for_GS_opacity=False,native_RGB_premultiplied_by_generated_alpha=False,
  selected_same_UV_primary_vs_GLB_RGB_abs_max=float(np.max(np.abs(sampled[:,:3]-glb_sample[:,:3]))),selected_same_UV_primary_vs_GLB_RGB_MAE=float(np.mean(np.abs(sampled[:,:3]-glb_sample[:,:3]))),
  texture_PNG_vs_GLB_RGB_abs_max=float(np.max(np.abs(raw[:,:,:3]-glb_raw[:,:,:3]))),texture_PNG_vs_GLB_RGB_MAE=float(np.mean(np.abs(raw[:,:,:3]-glb_raw[:,:,:3]))),
  query_primary_PNG_reproduction_max=float(np.max(np.abs(sampled-node['basecolor_rgba'][selected]))),
  bilinear_vs_trimesh_nearest_on_same_arbitraryUV_max=float(np.max(np.abs(sampled-nearest))),
  UV_policy='official output/Trimesh V-up to image top-left: x=u*W-.5, y=(1-v)*H-.5; texel-center bilinear/clamp. Trimesh uv_to_color uses round(u*(W-1)), round((1-v)*(H-1)) nearest; differs by kernel at arbitrary UV, matches exactly at texel centers',
  GLB_export='official output.export(extension_webp=True); embedded WebP may differ from primary lossless PNG. Native field uses primary PNG, not compressed GLB appearance',
  color_semantics='official PBR baseColor texture RGB interpreted under glTF sRGB convention as native sRGB color prior; no EOTF/radiance conversion/exposure fit. Intrinsic albedo/lighting/PBR truth is unidentified',
  no_true_white_paint_GT_claim=True,geometry_or_coverage_modified=False,new_inference_calls=0,GPU_operations=0,optimizer_updates=0,human_verdict=None)
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print(json.dumps(receipt),flush=True)
 if not all(checks.values()):raise SystemExit(1)
if __name__=='__main__':main()
