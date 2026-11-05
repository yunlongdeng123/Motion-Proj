"""Read-only preview ledger for fixedSAM/topology and native matte-only control."""
import hashlib,json,shutil
from pathlib import Path
BASE=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a6_insert_artifacts_20261105')
ROOT=BASE/'vehicle/matte_contract'
shutil.copyfile(BASE/'source/vehicle/matte_contract/architecture.mmd',ROOT/'architecture.mmd')
inventory=dict(status='EXISTING_WEIGHTS_USED_OFFLINE',local_common_HF_torch_Models_and_workspace_SAM_weights_found=False,
    remote_SAM2_environment='/root/autodl-tmp/envs/worldsim-v77-sam2',
    SAM2_source='/root/autodl-tmp/third_party/worldsim_v32/sam2',
    used_checkpoint='/root/autodl-tmp/third_party/worldsim_v32/sam2/checkpoints/sam2.1_hiera_large.pt',
    used_checkpoint_sha256='2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318',
    other_existing_weights=['/root/autodl-tmp/drivingeditor_repro/weights/sam_vit_h_4b8939.pth',
        '/root/autodl-tmp/models/worldsim_v77_sam3/sam3.pt','/root/autodl-tmp/models/idsplat/sam3/sam3.pt'],
    downloads=0,SAM_predict_calls=1,model_weight_updates=0,extra_filled_topology_SAM_calls=0,
    final_contract='matte_texture_samples_filled_v2.npz; rawSAM v1 matte-only is retained as causal diagnostic',
    final_colour_merge='New continuous surface must start from prototype initial RGB then40BUILD recolour; old formal protection is only this safety control')
(ROOT/'source_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
files=[]
for p in ROOT.rglob('*'):
    if not p.is_file() or p.name in ('preview_manifest.json','teacher_rgb_512.reencoded.png'):continue
    if p.name.endswith('_hr.png'):continue
    files.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(ROOT/'preview_manifest.json').write_text(json.dumps(dict(files=sorted(files,key=lambda f:f['path'])),indent=2)+'\n')
print(json.dumps(dict(files=len(files))))
