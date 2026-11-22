"""Record immutable source-mask resolution and observed source-image facts, CPU only."""
from pathlib import Path
import hashlib
import json
import numpy as np

WORK=Path(__file__).resolve().parents[2]
ROOT=WORK/'old_raw_teacher_joint_20261123_r9';AUDIT=ROOT/'audit'
C='/root/autodl-tmp/drivingeditor_repro/campaign'


def pin(path):
    p=Path(path);return {'path':p.resolve().as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def write(path,value):
    assert not path.exists(),f'will not overwrite {path}'
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


manifest=AUDIT/'raw_teacher_sources.json';mapping=AUDIT/'brown_mapping_audit.json'
rows=[]
for raw,expected in [(86,'cfc64974036c6bdad60655da309a5c48761bb32c92701fb92112155407c3760e'),(110,'e75f1684b3805952cfc8c877a6f6722928e48b1041f4d0db74c8272c27c59895')]:
    d=AUDIT/'source_inputs'/f'{raw:03d}';download=d/'R8_remote_rectified_masks.npz'
    local=WORK/'road_vehicle_joint_dggt_20261123_r8/audit/mask_coordinate_audit/rectified_masks'/f'{raw:03d}'/'rectified_masks.npz'
    a=np.load(local,allow_pickle=False);b=np.load(download,allow_pickle=False)
    equal={k:bool(np.array_equal(a[k],b[k])) for k in a.files}
    assert all(equal.values()) and pin(download)['sha256']==expected
    rows.append({'raw':raw,'resolved_remote_R8_mask':{
        'path':f'{C}/road_vehicle_joint_dggt_20261123_r8/audit/rectified_masks_v1/{raw:03d}/rectified_masks.npz',
        'sha256':expected,'bytes':download.stat().st_size,'actual_remote_readback_verified':True},
        'remote_bytes_local_copy':pin(download),'original_local_layout':pin(local),
        'all_arrays_bitexact':equal,'different_NPZ_bytes_same_arrays':True,
        'protected_mask_excludes_target_A20_by_design':True,
        'target_actor_visible_pixels':int(b['actor_visible'].sum()),
        'target_actor_visible_proxy_pixels':int(b['actor_visible_proxy'].sum()),
        'target_unknown_potential_pixels':int(b['unknown_potential'].sum()),
        'non_target_protected_pixels':int(b['protected'].sum()),
        'original_photo_semantic_proxy':{**pin(d/'original_photo_semantics_Brown.png'),
          'remote_path':f'/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/semantics/{raw:03d}.png',
          'domain':'Brown native 1920x1280','labelmap':'Cityscapes19 train IDs; road0, car13, vegetation8, sky10',
          'source':'pre-existing frozen SegFormer B5 applied to original real photo',
          'is_generated_teacher_semantics':False,'is_GT':False},
        'source_photo_local':pin(d/'original_photo_Brown.png'),
        'observation_json_local':pin(d/'observation.json')})
resolution={'schema':'R9.source_path_resolution.v1','complete':True,'source_manifest':pin(manifest),
  'original_manifest_unchanged':True,'reason':'Original manifest deliberately recorded nonexistent remote local-layout reference as unbound. This supplement binds the actual remote rectified_masks_v1 files.',
  'rows':rows,'CPU_only':True,'new_GPU_forward':0,'new_diffusion_calls':0,'parameter_updates':0}
write(AUDIT/'raw_teacher_source_path_resolution.json',resolution)

review={'schema':'R9.raw_teacher_visual_source_review.v1','complete':True,
  'reviewer':'independent assistant visual inspection of exact raw_model and mapped pinhole previews',
  'source_manifest':pin(manifest),'coordinate_audit':pin(mapping),
  'selection':{'frames':[86,110],'prompt_id':'legacy','seed':101,'post_hoc_seed_search':False},
  'decision':'SOURCE_CANDIDATES_REQUIRE_SURFACE_QUALIFICATION',
  'whole_image_road_teacher_accepted':False,'shared_surface_acceptance':'NOT_EVALUATED',
  'new_Gaussian_fit_authorized_by_this_review':False,
  'frames':[
    {'raw':86,'raw_model':pin(AUDIT/'source_inputs/086/raw_model.png'),
     'notes':['Road texture is continuous at full-view scale without the regular R7 quilt/teacher lattice.',
       'Central A20 black pickup, nearby vehicles, lane markings, real shadows, and background remain visible.',
       'The new R8 protected overlay intentionally omits the target A20 body. Its visible/unknown masks and source-specific road qualification must be handled separately.',
       'No whole-frame road label is valid for this output. Inspect only independently certified world-surface support after foreground exclusion.']},
    {'raw':110,'raw_model':pin(AUDIT/'source_inputs/110/raw_model.png'),
     'notes':['Lower road texture and lighting are legible; no regular R7 teacher lattice is apparent at full-view scale.',
       'Central A20 black pickup body and bumper remain in the original raw output, along with the white pickup and traffic.',
       'A few black pixels at the pinhole preview border are invalid inverse-warp support, not demonstrated road material or Gaussian alpha loss.',
       'Source-specific road qualification is needed for generated pixels; original-photo semantics alone cannot certify generated contours.']}
  ],'hard_final_role':'audit-only composite with real photo outside old W and inside old K; not RGB supervision',
  'pinhole_previews_role':'explicit coordinate conversion for review; raw_model remains canonical source',
  'CPU_only':True,'new_GPU_forward':0,'new_diffusion_calls':0,'training_updates':0}
write(AUDIT/'visual_source_review.json',review)

svg='''<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="320" viewBox="0 0 1120 320">
<rect width="1120" height="320" fill="#111827"/><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#93c5fd"/></marker></defs>
<g fill="#1f2937" stroke="#64748b"><rect x="20" y="48" width="205" height="100" rx="8"/><rect x="278" y="48" width="235" height="100" rx="8"/><rect x="566" y="48" width="245" height="100" rx="8"/><rect x="864" y="48" width="235" height="100" rx="8"/><rect x="278" y="208" width="533" height="80" rx="8"/></g>
<g fill="#e5e7eb" font-family="Arial,Microsoft YaHei,sans-serif" text-anchor="middle" font-size="17"><text x="122" y="80">原 raw 教师 · 086 / 110</text><text x="122" y="108">Brown · 1152×768</text><text x="122" y="132">legacy · seed101</text><text x="395" y="82">真实 XYZ → 相机 → Brown</text><text x="395" y="110">sRGB 解码后双线性采样</text><text x="688" y="82">同曲面可见性 + 前景排除</text><text x="688" y="110">UV / 深度 / 新保护 mask</text><text x="982" y="82">共享图联合校色与融合</text><text x="982" y="110">曲面验收后才进入 Gauss</text><text x="544" y="239">独立审计：源 SHA · 全 XYZ · 相机映射 · 分域预览</text><text x="544" y="267">hard final 仅作审计对照</text></g>
<g fill="none" stroke="#93c5fd" stroke-width="2.5" marker-end="url(#a)"><path d="M225 98H275"/><path d="M513 98H563"/><path d="M811 98H861"/><path d="M395 208V152"/><path d="M688 208V152"/></g></svg>'''
(AUDIT/'architecture_components.svg').write_text(svg,encoding='utf-8')
readme='''# R9 原始教师来源与坐标审计

固定 086、110 两个视角，legacy 提示词、seed101。两份 1152×768 raw_model 与旧验收 SHA 完全一致，原图域为 Brown 畸变图像整幅缩放；未重新生成、未选新 seed。

![Architecture components](architecture_components.svg)

canonical RGB 仅取 raw_model。raw_native 为其双线性放大；hard final 在旧 W 外和旧 K 内复制真实照片，两者仅为审计对照。两帧原始照片和 obs 已取回 source_inputs。

`raw_teacher_sources.json` 固定原始来源。`raw_teacher_source_path_resolution.json` 补绑远端 R8 实际 masks 路径；本地和远端 NPZ 的所有数组逐元素一致，压缩文件字节不同，分别保留 SHA。

`brown_mapping_audit.json` 通过解析 Brown / 缓存逆图 / 世界点直接投影三路核对。sRGB 先解码，再做线性域双线性插值。`mapping` 内 PNG 明示针孔预览；原 Brown 图保留于 source_inputs。

`fixed_surface_mesh_full_XYZ_audit.json` 使用固定网格完整 XYZ 按原 triangle rows/weights 重建，五帧与 source world_xyz 在 XYZ 三轴完全一致，重投影最高约 2e-10 像素。这不把所有世界中心自动判为可见，融合仍需同 UV / 深度一致门。

`visual_source_review.json` 记录原 raw 内仍存在中央 A20 车辆，R8 protected 特意不保护目标 actor。道路候选应按新保护、actor 可见性和来源自身道路资格排前景，不能把整图当路面。共享图与 Gaussian 在本审计中未执行。
'''
(AUDIT/'README.md').write_text(readme,encoding='utf-8')
print(json.dumps({n:pin(AUDIT/n) for n in ['raw_teacher_source_path_resolution.json','visual_source_review.json','architecture_components.svg','README.md']}))
