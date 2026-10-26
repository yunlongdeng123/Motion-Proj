"""打包输入准入、运动准备和撤源组件的必要证据，不打包传感器大文件/权重。"""
from pathlib import Path
import json,shutil,tarfile

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
DEST=ROOT/'admission_delivery_v2';DEST.mkdir(exist_ok=True)
for name in ['registry.json','operation_window_proposals.json','status.json']:
    shutil.copy2(ROOT/'evaluation_admission_v2'/name,DEST/name)
for name in ['ADMISSION_POLICY.json','admission_architecture.svg','EVALUATION_PROTOCOL.md']:
    shutil.copy2(ROOT/'code'/name,DEST/name)
for scene in (ROOT/'evaluation_admission_v1/scenes').iterdir():
    target=DEST/'contacts';target.mkdir(exist_ok=True)
    shutil.copy2(scene/'input_contact.jpg',target/(scene.name+'.jpg'))
for scene in (ROOT/'motion_scene_inputs_v1').iterdir():
    if not scene.is_dir():continue
    target=DEST/'motion_inputs'/scene.name;target.mkdir(parents=True,exist_ok=True)
    shutil.copy2(scene/'sensor_preparation.json',target/'sensor_preparation.json')
    shutil.copytree(scene/'projection_QA',target/'projection_QA',dirs_exist_ok=True)
    prep=scene/'native_preparation'
    if prep.exists():
        for p in prep.iterdir():
            if p.suffix in ['.json','.jpg']:shutil.copy2(p,target/p.name)
for run in (ROOT/'native_scene_training_v1').iterdir():
    target=DEST/'native_training'/run.name;target.mkdir(parents=True,exist_ok=True)
    for p in run.rglob('*.json'):
        if p.name in ['status.json','smoke_gate.json','latest_complete_node.json','returned.json','restore_verification.json','cuda_memory_peak.json','config_delta.json','actual_scene_binding.json']:
            q=target/p.relative_to(run);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
for name in ['scene0017','scene0029']:
    source=ROOT/'evaluation_withdrawal_v2'/name;target=DEST/'withdrawal'/name;target.mkdir(parents=True,exist_ok=True)
    for f in ['summary.json','comparison.jpg','reference_layers.png','summary_before_png_alignment.json']:
        shutil.copy2(source/'measurement_evaluation'/f,target/f)
    shutil.copy2(source/'propainter/summary.json',target/'backend_summary.json')
    shutil.copy2(source/'propainter/video.mp4',target/'generated.mp4')
for name in ['motion_masks_status.json','motion_masks_dispatch.json','native_scene_dispatch.json','quota_state.json','phase2_git_publish.json']:
    p=ROOT/name
    if p.exists():shutil.copy2(p,DEST/name)
archive=ROOT/'admission_delivery_v2.tar.gz'
with tarfile.open(archive,'w:gz') as tar:tar.add(DEST,arcname='admission',recursive=True)
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size}))
