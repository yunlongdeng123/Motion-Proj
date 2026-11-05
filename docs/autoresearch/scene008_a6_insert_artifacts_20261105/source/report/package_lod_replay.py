"""Small S/F replay bundle; original large native scene stays a declared dependency."""
from pathlib import Path
import hashlib,json,shutil,zipfile
ROOT=Path(__file__).resolve().parents[3]
ART=ROOT/'work/a6_artifact_diagnosis'
DEST=ROOT/'outputs/scene008_vehicle_appearance/diagnosis/replay_bundle'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    DEST.mkdir(parents=True,exist_ok=True)
    files={
        'complete_a6.npz':ART/'surface_fit/combined_v1/observed_fit/formal/complete_a6.npz',
        'posterior.npz':ART/'surface_fit/results/posterior.npz',
        'material_SH_contract.json':ART/'scale/lod_material_SH_contract.json',
        'threeview_receipt.json':ART/'scale/lod_receipt.remote.json'}
    for name in ('surface_lod_asset.py','replay_surface_lod.py','probe_surface_lod.py','relative_area_asset.py','scale_probe.py'):
        files[name]=ART/'scale'/name
    smoke=ART/'scale/lod_replay_smoke_receipt.json'
    if smoke.is_file():files['replay_smoke_receipt.json']=smoke
    for name,p in files.items():
        assert p.is_file(),p
        q=DEST/name
        if not q.is_file() or sha(q)!=sha(p):shutil.copy2(p,q)
    command='''# Extract this bundle into a NEW directory on the existing AutoDL host.
# Run from that directory. Choose an output path that does not yet exist.
PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/envs/drivingeditor/bin/python replay_surface_lod.py \\
  --asset complete_a6.npz --surface posterior.npz --raw 113 --width 480 \\
  --output /root/autodl-tmp/drivingeditor_repro/campaign/scene008_a6_insert_artifacts_20261105/user_lod_replay_113
'''
    (DEST/'REPLAY_COMMAND.txt').write_text(command,encoding='utf-8')
    manifest=dict(status='PERSISTENT_SURFACE_MATERIAL_PLUS_VIEWPORT_SAMPLER_BUNDLE',
        canonical_NPZ_alone_uses_dense_default_rows=True,naked_NPZ_does_not_reproduce_LOD=True,
        canonical_asset_sha256=sha(DEST/'complete_a6.npz'),posterior_sha256=sha(DEST/'posterior.npz'),
        persistent_S='posterior cubic depth coefficients, knots, reference_K and reference_camera_from_local_m',
        persistent_F='final canonical SH0 RGB and chart opacity/.99 on resample_reference_pixel_xy; outside zero alpha',
        material_filter='Fixed 2x2 area mip pyramid of premultRGB+coverage; bilinear per level then trilinear; unpremultiply valid coverage',
        view_sampling='du=1/sqrt(median(abs(det d(screenXY)/d(UV)))); UV=(integer+.5)*du-.5; sigma=.6*du*S_z/reference_fx',
        opacity='Peak .99 times filtered material coverage, no determinant-area attenuation',
        retained_source_rows=2778,retained_all_parameters_unchanged=True,
        edit_states_share_same_SF_and_camera_rows=True,screen_space_overlay=False,
        source_surface_is_reference106_conditioned=True,complete360_or_hidden_geometry_accuracy_claim=False,
        nonDC_material_contract='Canonical declares SH3 but actual all/chart/retained nonDC terms are exact0; wrapper refuses nonzero nonDC inputs',
        existing_host_dependencies={
            'ssh_alias':'wm-vgpu-1011',
            'python':'/root/autodl-tmp/envs/drivingeditor/bin/python',
            'Runtime_and_SHAsset_source':'/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a6_asset_controls_20261105/source',
            'engine':'/root/autodl-tmp/drivingeditor_repro/campaign/r15_baseline_owner5_20261018/training_source/scene0008_089_137',
            'frozen_scene_observations':'/root/autodl-tmp/drivingeditor_repro/campaign/scene008_instances_20261105_r1/observations/frames',
            'native_parent':'/root/autodl-tmp/drivingeditor_repro/campaign/scene008_native_ownership_20261105/run_r3/formal',
            'road_checkpoint':'/root/autodl-tmp/drivingeditor_repro/campaign/scene008_native_road1k_20261105_r1/formal/checkpoint001000.pt',
            'fine_road':'/root/autodl-tmp/drivingeditor_repro/campaign/scene008_ray_road_colour_fine_20261105',
            'active_SO_SHA256':'0eaff4d7a9b8ef6e711627f4c19824bf427c4387f7c6453fcb769a2c919c920e',
            'legacy_neg_fov':True,'shared_SO_modified':False},
        software='Use the existing torch/numpy/scipy/Pillow/native CUDA environment; no model download or weight update.',
        command_file='REPLAY_COMMAND.txt',files={name:dict(bytes=(DEST/name).stat().st_size,sha256=sha(DEST/name)) for name in files},
        automatic_adoption=False)
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    text='''LOD replay package

complete_a6.npz is the canonical persistent sampled asset. Loading it with its
ordinary .rows keeps the dense default raster; that alone does not give the LOD
screenshots. SurfaceLODAsset combines its fixed final material F with posterior
surface S and the camera-frequency sampler. The current camera changes render
sampling density/filtering; keep/delete never changes S/F or the asset rows for
that same camera. The 2778 retained PVD rows are carried without modification.

The bundle deliberately excludes the large frozen native scene/checkpoints and
CUDA environment. Their existing AutoDL absolute dependencies are in manifest.
REPLAY_COMMAND.txt produces solo, keep and delete native images + raw floats and
a receipt in a fresh output directory. No training is involved. Three-view and
replay smoke receipts state the actual validation; this is a research rendering
candidate, not an adoption or 360-degree/hidden-geometry accuracy assertion.
'''
    (DEST/'README.txt').write_text(text,encoding='utf-8')
    archive=DEST.parent/'surface_LOD_replay_bundle.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(DEST.iterdir()):
            if p.is_file():z.write(p,p.name)
    print(json.dumps(dict(ready=True,bundle=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),smoke_included=smoke.is_file())))

if __name__=='__main__':main()
