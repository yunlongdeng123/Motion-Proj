"""打包输入准入、运动准备和撤源组件的必要证据，不打包传感器大文件/权重。"""
from pathlib import Path
import json,shutil,tarfile

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
DEST=ROOT/'admission_delivery_v2';DEST.mkdir(exist_ok=True)
for name in ['registry.json','operation_window_proposals.json','status.json']:
    shutil.copy2(ROOT/'evaluation_admission_v2'/name,DEST/name)
for name in ['ADMISSION_POLICY.json','admission_architecture.svg','EVALUATION_PROTOCOL.md','motion_diagnostics_architecture.svg']:
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
    for event in (run/'formal30k/checkpoint_events').glob('*.json'):
        d=json.loads(event.read_text());step=d['iteration'];node=target/f'checkpoint{step}';node.mkdir(exist_ok=True)
        for k in ['metrics','paper_metrics']:
            if d.get(k):shutil.copy2(d[k],node/(k+'.json'))
        shutil.copy2(event,node/'event.json')
        # Save exact native PNGs for a declared middle camera; no rerendering.
        for channel,path in d['frames'][24]['channels'].items():shutil.copy2(path,node/(channel+'.png'))
for suite in (ROOT/'motion_edit_suite_v1').iterdir():
    if suite.is_dir():shutil.copytree(suite,DEST/'motion_requests'/suite.name,dirs_exist_ok=True)
    elif suite.suffix=='.json':
        (DEST/'motion_requests').mkdir(exist_ok=True);shutil.copy2(suite,DEST/'motion_requests'/suite.name)
for p in (ROOT/'motion_suite_execution_v1').glob('*/status.json'):
    target=DEST/'motion_execution'/p.parent.name;target.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target/'status.json')
    suite=p.parent/'native_suite'
    if (suite/'summary.json').exists():shutil.copy2(suite/'summary.json',target/'native_summary.json')
    for case in suite.iterdir() if suite.exists() else []:
        if not case.is_dir():continue
        frames=sorted((case/'native_rgb').glob('*.png'))
        if not frames:continue
        middle=frames[len(frames)//2];q=target/case.name;q.mkdir(exist_ok=True)
        shutil.copy2(middle,q/'native.png')
        for source,name in [(case/'mask'/middle.name,'mask.png'),(case/'propainter/model_png'/middle.name,'generated.png'),(case/'propainter/summary.json','backend_summary.json')]:
            if source.exists():shutil.copy2(source,q/name)
for name in ['scene0017','scene0029']:
    source=ROOT/'evaluation_withdrawal_v2'/name;target=DEST/'withdrawal'/name;target.mkdir(parents=True,exist_ok=True)
    for f in ['summary.json','comparison.jpg','reference_layers.png','summary_before_png_alignment.json']:
        shutil.copy2(source/'measurement_evaluation'/f,target/f)
    shutil.copy2(source/'propainter/summary.json',target/'backend_summary.json')
    shutil.copy2(source/'propainter/video.mp4',target/'generated.mp4')
for name in ['motion_masks_status.json','motion_masks_dispatch.json','native_scene_dispatch.json','native_scene_priority_dispatch.json','motion_suite_dispatch.json','native_actor_source_audit.json','motion_suite_engineering_ledger.json','quota_state.json','phase2_git_publish.json']:
    p=ROOT/name
    if p.exists():shutil.copy2(p,DEST/name)
archive=ROOT/'admission_delivery_v2.tar.gz'
with tarfile.open(archive,'w:gz') as tar:tar.add(DEST,arcname='admission',recursive=True)
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size}))
