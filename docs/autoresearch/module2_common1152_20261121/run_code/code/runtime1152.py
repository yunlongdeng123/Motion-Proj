"""Frozen geometry and real 1152 camera factory for the new teacher refit."""
from pathlib import Path
import hashlib, json, sys, time
import numpy as np

C = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ROOT = C / 'road_common1152_20261121'
Q = C / 'road_teacher_detail_cpu_20261119_r1'
G = Q / 'gpu_20261119'
B = G / 'native_capacity_20261120_r1'
REPO = Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
ASSET = C / 'road_activation_20261118_r1/B32_beam95_native_r1/asset_guarded'
FRAMES = (86, 110, 134, 179, 198)
OWNER = 'road_common1152'

def write(p, d):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    q.replace(p)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for s in iter(lambda:f.read(4*1024*1024),b''): h.update(s)
    return h.hexdigest()

def load(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}

def paths():
    sys.path[:0]=[str(C/'road_capacity_gpu_20261118_r1/code'),str(REPO/'scripts/worldsim_v82'),
                  str(Q/'code'),str(G/'field_training/code'),str(B/'code'),
                  str(B/'shared_field/code'),str(B/'gs_oracle/code')]

def camera_headers():
    paths()
    from drivingeditor_dynamic_edit.v44_manifest import SceneManifest
    source=ASSET/'persistent/scene.json'; data=json.loads(source.read_text())
    data['render_wh']=[1152,768]
    m=SceneManifest(data,source.parent)
    headers=m.camera_headers()
    rows=[]
    for raw in FRAMES:
        h=headers[raw]
        rows.append({k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in h.items()})
    write(ROOT/'camera_manifest.json',dict(schema='common1152_camera_headers.v1',resolution_wh=[1152,768],
        source_scene=str(source),source_scene_SHA256=sha(source),native_reference_wh=[1920,1280],
        intrinsics_scale_xy=[.6,.6],pixel_grid='same make_native_camera convention; no crop/letterbox',rows=rows))
    return headers

def context(frames=FRAMES):
    paths()
    from drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff
    from drivingeditor_dynamic_edit.v44_manifest import make_native_camera
    from road_capacity_native import NativeRoadContext
    import torch
    headers=camera_headers(); cameras={}
    scene,m,loader=load_persistent_handoff(ASSET/'persistent/scene.json',ASSET/'persistent/proof.json')
    def camera(raw):
        if raw not in cameras: cameras[raw]=make_native_camera(headers[raw])
        return cameras[raw]
    scene.camera_provider=camera
    for component in (scene.static,scene.dynamic,scene.environment):
        for p in component.parameters() if hasattr(component,'parameters') else (): p.requires_grad_(False)
    ctx=NativeRoadContext(scene,m,loader,ASSET,frames,ROOT/'gpu_lease.json',OWNER)
    ctx._manifest_path=ASSET/'persistent/scene.json'; ctx._proof_path=ASSET/'persistent/proof.json'
    assert all(ctx.camera(r).image_width==1152 and ctx.camera(r).image_height==768 for r in frames)
    return ctx

def status(phase, **kw):
    p=ROOT/'status.json'
    old=json.loads(p.read_text()) if p.exists() else {}
    old.update(schema='common1152_status.v1',status='running',state='running',phase=phase,updated_at=time.time(),**kw)
    write(p,old)
