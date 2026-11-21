"""Freeze the accepted 1152 teacher and CPU source permissions, without inference."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import hashlib, json, shutil, sys
import numpy as np
from PIL import Image

BASE = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
P = BASE / 'road_2d_teacher_prompt_20261120_r1'
Q = BASE / 'road_2d_teacher_hf_cpu_20261119_r1'
OUT = BASE / 'road_common1152_20261121/teacher'
sys.path.insert(0, str(Q / 'code'))
from teacher_supervision_view_cpu import build_view_supervision, fixed_view_spec

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):
    p = Path(p)
    return dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size)
def check(rec):
    assert Path(rec['path']).is_file() and sha(rec['path']) == rec['sha256']
    return Path(rec['path'])
def image(p): return np.asarray(Image.open(p).convert('RGB')).copy()
def write(p, x): Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n')

assert not (OUT / 'manifest.json').exists(), 'Preserve frozen output'
OUT.mkdir(parents=True, exist_ok=True)
assets = read(P / 'report_assets_r1/asset_manifest.json')
pred_path = check(assets['predictions'])
pred = read(pred_path)
plan_path = check(pred['manifest'])
plan = read(plan_path)
defaults = read(P / 'teacher_defaults.json')
assert defaults['selected_arm'] == 'full1152_T0'
assert defaults['prompt_policy']['active_prompt_id'] == 'legacy'
manifest = dict(schema='road_common1152_teacher_inputs.v1', complete=False,
    output_wh=[1152,768], train_frames=[86,110], diagnostic_frames=[134,179,198],
    selection=dict(arm='full1152_T0', prompt='legacy', seed=101,
        model='SD1.5 inpainting + attention LoRA T0 step600',
        rule='Existing report default_seed=101; fixed before student fitting; no best-of-N selection'),
    inference_calls=0, training_updates=0, renderer_calls=0, GPU_used=False,
    sources=dict(defaults=pin(P/'teacher_defaults.json'), predictions=pin(pred_path),
        generation_plan=pin(plan_path), source_code=pin(Path(__file__)),
        domain_builder=pin(Q/'code/teacher_supervision_view_cpu.py')),
    NPZ_schema=dict(target_A_srgb='float32 CHW [0,1], original observation one AREA sample',
        target_C_srgb='float32 CHW [0,1], byte-exact 1152 generated RGB /255',
        target_S_srgb='same exact generated RGB as C; only S mask is supervised',
        A_C_S='mutually exclusive bool HW, full native footprint and source support',
        K='bool HW; any retained-actor native support excludes',
        protected='bool HW; extra forbidden domain',
        teacher_write_full='bool HW; all native support within actual W_output, not H_input',
        q_A_2d_q_C_2d_q_S_2d='float32 HW binary source eligibility, not learned confidence',
        camera_K='float64 3x3, legacy half-pixel scaling; renderer must independently qualify'),
    S_rule='Frozen high & actual W_output & ~uncertain & ~boundary & association/uncertainty gates & ~other_shadow & ~K; same generated model output, not whole-frame shadow supervision',
    qualification=dict(CPU_source_permissions=True, raster_camera_verified=False,
        requires_before_training='Intersect source masks with new fixed geometry/actual alpha/retained-actor qualifications; do not treat q_2d as 3D confidence'),
    frames={})
for raw in (86,110):
    sid=f'scene065_raw{raw:03d}'
    case=next(c for c in plan['cases'] if c['id']==sid)
    job=next(j for j in pred['jobs'] if j['case_id']==sid and j['arm']=='full1152_T0' and j['prompt_id']=='legacy' and j['seed']==101)
    assert job['historical_replay_exact']['raw_model'] and job['W_outside_RGB_exact'] and job['K_RGB_exact']
    raw_path=check(job['raw_model']); observed_path=check(case['observation'])
    role_path=check(case['roles_native']); model_role_path=check(case['view']['roles_model'])
    construction_path=check(case['view']['construction'])
    old=Q/f'supervision_gpu_r1/raw{raw:03d}/full1152_T0_seed101/1152x768'
    config=read(old/'rebuild_config.json')
    domains_path=Path(config['native_domains'])
    with np.load(domains_path,allow_pickle=False) as z: domains={k:z[k].copy() for k in z.files}
    with np.load(role_path,allow_pickle=False) as z: roles={k:z[k].copy() for k in z.files}
    assert np.array_equal(domains['K'],roles['K'])
    assert not np.any((domains['C']|domains['S']) & ~roles['W_output'])
    old_write=domains['teacher_write'].copy()
    domains['teacher_write'] &= roles['W_output']
    rgb=image(raw_path); obs=image(observed_path)
    assert rgb.shape==(768,1152,3) and obs.shape==(1280,1920,3)
    spec=fixed_view_spec('full1152')
    arrays,report=build_view_supervision(rgb,obs,domains,
        source_model_valid=np.ones((768,1152),bool),
        native_window_xyxy=spec['native_window_xyxy'], native_from_model=spec['native_from_model'],
        output_wh=(1152,768),intrinsics=config['camera']['intrinsics'],camera_wh=(1920,1280),
        camera_convention=config['camera']['convention'],source_kind='lossless_raw_model',view_name='full1152')
    assert np.array_equal(arrays['target_C_srgb'],rgb.transpose(2,0,1).astype(np.float32)/255)
    assert np.array_equal(arrays['target_C_srgb'],arrays['target_S_srgb'])
    assert all(not np.any(arrays[a]&arrays[b]) for a,b in [('A','C'),('A','S'),('C','S')])
    assert not np.any((arrays['A']|arrays['C']|arrays['S']) & (arrays['K']|arrays['protected']))
    assert not np.any((arrays['C']|arrays['S']) & ~arrays['teacher_write_full'])
    folder=OUT/f'raw{raw:03d}'; folder.mkdir(exist_ok=False)
    np.savez_compressed(folder/'supervision.npz',**arrays)
    np.savez_compressed(folder/'native_domains.npz',**domains)
    shutil.copy2(raw_path,folder/'teacher_rgb.png')
    shutil.copy2(observed_path,folder/'observation_native.png')
    shutil.copy2(role_path,folder/'roles_native.npz')
    shutil.copy2(model_role_path,folder/'roles_model.npz')
    shutil.copy2(construction_path,folder/'construction.json')
    Image.fromarray(np.rint(arrays['target_A_srgb'].transpose(1,2,0)*255).astype(np.uint8)).save(folder/'observation_1152.png')
    final=np.rint(arrays['target_A_srgb'].transpose(1,2,0)*255).astype(np.uint8)
    final[arrays['teacher_write_full']]=rgb[arrays['teacher_write_full']]
    Image.fromarray(final).save(folder/'teacher_legal_preview.png')
    for key in ('A','C','S','K','teacher_write_full','protected'):
        Image.fromarray(arrays[key].astype(np.uint8)*255).save(folder/f'{key}.png')
    camera=dict(raw=raw,scene='Scene065',camera='FRONT',native_wh=[1920,1280],output_wh=[1152,768],
        crop_or_letterbox=False,native_window_xyxy=[0,0,1920,1280],
        intrinsics_native=config['camera']['intrinsics'],intrinsics_1152=arrays['camera_K'].tolist(),
        native_from_model=spec['native_from_model'],
        convention='pixel-cell edges; index+.5 centers; K scales with legacy half-pixel convention',
        raster_camera_verified=False,pose_binding='Scene065 FRONT camera for the same original raw index; rendering agent must confirm original image/pose',
        observation_source=pin(observed_path))
    write(folder/'camera.json',camera); write(folder/'qualification.json',report)
    manifest['frames'][str(raw)]=dict(raw=raw,job_id=job['id'],sources=dict(raw_model=pin(raw_path),
        observation=pin(observed_path),roles_native=pin(role_path),roles_model=pin(model_role_path),
        original_domains=pin(domains_path),construction=pin(construction_path),final_native_reference=job['final_native']),
        counts={k:int(arrays[k].sum()) for k in ('A','C','S','K','teacher_write_full')},
        stricter_actual_write_removed_native_pixels=int((old_write&~domains['teacher_write']).sum()),
        C_S_same_teacher=True,source_level_teacher_exact=True,camera=camera,
        files={f.name:pin(f) for f in sorted(folder.iterdir())})
    print(json.dumps(dict(raw=raw,path=str(folder),counts=manifest['frames'][str(raw)]['counts'])),flush=True)
manifest['complete']=True
write(OUT/'manifest.json',manifest)
print(json.dumps(dict(manifest=str(OUT/'manifest.json'),sha256=sha(OUT/'manifest.json'),CPU_only=True)),flush=True)
