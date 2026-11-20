"""Explicit immutable data and true native1920 context factories; no auto-start."""
from pathlib import Path
import json,sys,hashlib
import numpy as np

CAMPAIGN=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
Q=CAMPAIGN/'road_teacher_detail_cpu_20261119_r1';G=Q/'gpu_20261119'
B=G/'native_capacity_20261120_r1';PREVIOUS=G/'quality_bench_20261120_r1'
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
ASSET=CAMPAIGN/'road_activation_20261118_r1/B32_beam95_native_r1/asset_guarded'
FRAMES=(86,110)

def load(p):
    with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def real_frames(split='seen'):
    if split not in('seen','held'):raise ValueError('registered split only')
    frames={};bindings={}
    for raw in FRAMES:
        tpath=PREVIOUS/f'highres/targets/{raw:03d}.npz';spath=B/f'qa/source_split_r1/{raw:03d}.npz'
        t,s=load(tpath),load(spath)
        assert np.array_equal(t['K_vehicle'],s['K'])
        frames[raw]=dict(alpha=s['fixed_alpha'],targets={k:t['T_'+k]for k in('A','C','S')},
            contexts={k:s[k+'_'+split+'_context']for k in('A','C','S')},
            scores={k:s[k+'_'+split]for k in('A','C','S')})
        bindings.update({str(p):sha(p)for p in(tpath,spath)})
    return frames,dict(files=bindings,split=split,held_is_blind=False,
        prior='Common A200/A0 and routing were previously exposed to both views; no new held target label is used in current train loss or normalization.')

def make_context(lease_file,owner,frames=FRAMES):
    # Existing native helper validates the new caller-owned lease before scene load.
    sys.path[:0]=[str(CAMPAIGN/'road_capacity_gpu_20261118_r1/code'),str(REPO/'scripts/worldsim_v82')]
    from road_capacity_native import NativeRoadContext,assert_gpu_lease
    assert_gpu_lease(lease_file,owner)
    from drivingeditor_dynamic_edit.persistent_road_handoff import load_persistent_handoff
    from drivingeditor_dynamic_edit.v44_manifest import SceneManifest,make_native_camera
    manifest=ASSET/'persistent/scene.json';proof=ASSET/'persistent/proof.json'
    scene,m,receipt=load_persistent_handoff(manifest,proof)
    for c in(scene.static,scene.dynamic,scene.environment):
        for p in c.parameters()if hasattr(c,'parameters')else():p.requires_grad_(False)
    data=dict(m.data);data['render_wh']=[1920,1280]
    high=SceneManifest(data,manifest.parent).camera_headers();low=m.camera_headers();cameras={}
    for raw in frames:
        assert np.array_equal(low[raw]['W'],high[raw]['W'])
        assert all(high[raw][k]==4*low[raw][k]for k in('fx','fy','cx','cy'))
    def camera(raw):
        if raw not in cameras:cameras[raw]=make_native_camera(high[raw])
        return cameras[raw]
    scene.camera_provider=camera
    ctx=NativeRoadContext(scene,m,receipt,ASSET,frames,lease_file,owner)
    ctx._manifest_path=manifest;ctx._proof_path=proof
    return ctx

def field_and_reader(parent_rgb,device='cuda'):
    from shared_field_gs_read import load_common_avg_model,SharedAverageGSRead
    field=G/'formal_r1/avg_common_A200/field.pt';permission=Q/'permission/final_r1/teacher_evidence.npz'
    runtime=Q/'code/world_texture_runtime.py'
    model,binding=load_common_avg_model(field,permission,runtime)
    ids=load(permission)
    reader=SharedAverageGSRead.from_operator(Q/'permission/avg_sparse_gh31_r1/operator',
        ids['active_world_rows'],ids['common_write_active'],parent_rgb.detach().cpu().numpy(),gate='teacher')
    return model.to(device),reader,dict(field=binding,read=reader.describe())

def cli_paths():
    sys.path[:0]=[str(B/'code'),str(B/'shared_field/code'),str(B/'gs_oracle/code')]
