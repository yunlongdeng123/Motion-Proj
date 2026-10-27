"""打包输入准入、运动准备和撤源组件的必要证据，不打包传感器大文件/权重。"""
from pathlib import Path
import argparse,hashlib,json,shutil,tarfile
parser=argparse.ArgumentParser();parser.add_argument('--delta-base',type=Path);args=parser.parse_args()
def digest_file(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
DEST=ROOT/'admission_delivery_v2';DEST.mkdir(exist_ok=True)
for name in ['registry.json','operation_window_proposals.json','status.json']:
    shutil.copy2(ROOT/'evaluation_admission_v2'/name,DEST/name)
for name in ['ADMISSION_POLICY.json','admission_architecture.svg','EVALUATION_PROTOCOL.md','motion_diagnostics_architecture.svg']:
    shutil.copy2(ROOT/'code'/name,DEST/name)
for name in ['BENCHMARK_SCALE.md','benchmark_expansion_architecture.svg','TIME_AND_SPACE_CONTRACT.md']:
    shutil.copy2(ROOT/'code'/name,DEST/name)
shutil.copy2(ROOT/'code/benchmark_execution_architecture.svg',DEST/'benchmark_execution_architecture.svg')
shutil.copy2(ROOT/'code/layer_diagnostics_architecture.svg',DEST/'layer_diagnostics_architecture.svg')
for folder in (ROOT/'native_layer_isolation_v1').glob('*'):
    if not folder.is_dir():continue
    target=DEST/'layer_isolation'/folder.name;target.mkdir(parents=True,exist_ok=True)
    for name in ['status.json','summary.json','cpu_summary.json','comparison_index23.jpg','comparison_index30.jpg']:
        if (folder/name).exists():shutil.copy2(folder/name,target/name)
for folder in (ROOT/'observed_pose_diagnostics_v3').glob('*'):
    if not folder.is_dir():continue
    target=DEST/'observed_pose'/folder.name;target.mkdir(parents=True,exist_ok=True)
    for name in ['summary.json','comparison.jpg']:
        if (folder/name).exists():shutil.copy2(folder/name,target/name)
for path in (ROOT/'benchmark_execution_v1').glob('*/status.json'):
    target=DEST/'benchmark_execution'/path.parent.name;target.mkdir(parents=True,exist_ok=True)
    shutil.copy2(path,target/'status.json')
    state=json.loads(path.read_text())
    for name,stage in state.get('stages',{}).items():
        summary=Path(stage['summary'])
        if summary.exists():shutil.copy2(summary,target/('stage_'+name+'_summary.json'))
    if (path.parent/'cpu_delivery/summary.json').exists():shutil.copytree(path.parent/'cpu_delivery',target/'cpu_delivery',dirs_exist_ok=True)
for name in ['benchmark_adapter_cpu_audit_v1.json','benchmark_execution_dispatch.json','ego_speed_clock_CPU_audit.json','ego_speed_clock_dispatch.json','benchmark_request_space_audit_v1.json','benchmark_asset_diversity_audit_v1.json']:
    if (ROOT/name).exists():shutil.copy2(ROOT/name,DEST/name)
sh_control=ROOT/'motion_SH_operator_control_v1'
if (sh_control/'summary.json').exists():
    target=DEST/'sh_operator_control';target.mkdir(exist_ok=True)
    shutil.copy2(sh_control/'summary.json',target/'summary.json')
    for name in ['cpu_control_summary.json','comparison.jpg']:
        if (sh_control/name).exists():shutil.copy2(sh_control/name,target/name)
    for case in sh_control.iterdir():
        frames=sorted((case/'native_rgb').glob('*.png')) if case.is_dir() else []
        if frames:shutil.copy2(frames[3*len(frames)//4],target/(case.name+'.png'))
for folder in (ROOT/'ego_speed_clock_control_v1').glob('*'):
    if not folder.is_dir():continue
    target=DEST/'ego_speed_clock_control'/folder.name;target.mkdir(parents=True,exist_ok=True)
    for name in ['status.json','summary.json','comparison.jpg']:
        if (folder/name).exists():shutil.copy2(folder/name,target/name)
    if (folder/'status.json').exists():
        state=json.loads((folder/'status.json').read_text());native=Path(state.get('native_output',str(folder/'fixed_world')))
        if (native/'summary.json').exists():shutil.copy2(native/'summary.json',target/'native_batch_summary.json')
        if (native/'ego_speed/native_rgb_h264.mp4').exists():shutil.copy2(native/'ego_speed/native_rgb_h264.mp4',target/'fixed_world_native.mp4')
ledger=ROOT/'benchmark_cases_v1/scale_ledger.json'
if ledger.exists():shutil.copy2(ledger,DEST/'benchmark_scale_ledger.json')
for source in (ROOT/'benchmark_cases_v1/scenes').glob('*/summary.json'):
    folder=source.parent;target=DEST/'benchmark_inputs'/folder.name;target.mkdir(parents=True,exist_ok=True)
    for name in ['summary.json','input_quality.json','case_inputs.jpg']:
        shutil.copy2(folder/name,target/name)
    shutil.copytree(folder/'requests',target/'requests',dirs_exist_ok=True)
    shutil.copy2(ROOT/'motion_scene_inputs_v1'/folder.name/'sensor_preparation.json',target/'sensor_preparation.json')
for scene in (ROOT/'evaluation_admission_v1/scenes').iterdir():
    target=DEST/'contacts';target.mkdir(exist_ok=True)
    shutil.copy2(scene/'input_contact.jpg',target/(scene.name+'.jpg'))
for scene in (ROOT/'motion_scene_inputs_v1').iterdir():
    if not scene.is_dir():continue
    if not (scene/'sensor_preparation.json').exists():continue
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
        if p.name in ['status.json','smoke_gate.json','latest_complete_node.json','returned.json','restore_verification.json','cuda_memory_peak.json','config_delta.json','actual_scene_binding.json','formal30k_agent_acceptance.json','checkpoint_retention.json']:
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
    state=json.loads(p.read_text());suite=Path(state.get('native_suite_path',str(p.parent/'native_suite')))
    assert suite.resolve().is_relative_to(p.parent.resolve())
    if (suite/'summary.json').exists():shutil.copy2(suite/'summary.json',target/'native_summary.json')
    if (suite/'cpu_diagnostics/summary.json').exists():shutil.copytree(suite/'cpu_diagnostics',target/'diagnostics',dirs_exist_ok=True)
    continuous=suite/'continuous'
    if (continuous/'summary.json').exists():
        q=target/'continuous';q.mkdir(exist_ok=True);shutil.copy2(continuous/'summary.json',q/'summary.json')
        audit=continuous/'fixed_backend_audit'
        replay=continuous/'fixed_backend_same_input_replay'
        if replay.exists():
            delivery=q/'same_input_replay';delivery.mkdir(exist_ok=True)
            for name in ['status.json','summary.json']:
                if (replay/name).exists():shutil.copy2(replay/name,delivery/name)
        if (audit/'summary.json').exists():
            delivery=q/'fixed_backend';delivery.mkdir(exist_ok=True)
            for name in ['status.json','summary.json','cpu_delivery_summary.json','comparison.jpg']:
                if (audit/name).exists():shutil.copy2(audit/name,delivery/name)
            for revision_name in ['delete_original','insert_reused','move_inserted','turn_inserted','revisit_original_camera']:
                revision=continuous/revision_name;frames=sorted((revision/'propainter/model_png').glob('*.png'))
                if frames:shutil.copy2(frames[3*len(frames)//4],delivery/(revision_name+'.png'))
                for source,name in [(revision/'propainter/model_png_h264.mp4',revision_name+'.mp4'),(revision/'ordinary_cache_receipt.json',revision_name+'_cache_receipt.json')]:
                    if source.exists():shutil.copy2(source,delivery/name)
        for revision in continuous.iterdir():
            frames=sorted((revision/'native_rgb').glob('*.png')) if revision.is_dir() else []
            if frames:shutil.copy2(frames[len(frames)//2],q/(revision.name+'.png'))
            if (revision/'native_rgb_h264.mp4').exists():shutil.copy2(revision/'native_rgb_h264.mp4',q/(revision.name+'.mp4'))
    for case in suite.iterdir() if suite.exists() else []:
        if not case.is_dir():continue
        frames=sorted((case/'native_rgb').glob('*.png'))
        if not frames:continue
        middle=frames[3*len(frames)//4];q=target/case.name;q.mkdir(exist_ok=True)
        (q/'preview_frame.json').write_text(json.dumps(dict(raw=int(middle.stem),selection='same floor(3N/4) index, avoiding speed identity anchor',source=str(middle)),indent=2))
        shutil.copy2(middle,q/'native.png')
        for source,name in [(case/'mask'/middle.name,'mask.png'),(case/'propainter/model_png'/middle.name,'generated.png'),(case/'propainter/summary.json','backend_summary.json')]:
            if source.exists():shutil.copy2(source,q/name)
        for source,name in [(case/'native_rgb_h264.mp4','native_h264.mp4'),(case/'propainter/model_png_h264.mp4','generated_h264.mp4')]:
            if source.exists():shutil.copy2(source,q/name)
for name in ['scene0017','scene0029']:
    source=ROOT/'evaluation_withdrawal_v2'/name;target=DEST/'withdrawal'/name;target.mkdir(parents=True,exist_ok=True)
    for f in ['summary.json','comparison.jpg','reference_layers.png','summary_before_png_alignment.json']:
        shutil.copy2(source/'measurement_evaluation'/f,target/f)
    shutil.copy2(source/'propainter/summary.json',target/'backend_summary.json')
    shutil.copy2(source/'propainter/video.mp4',target/'generated.mp4')
for name in ['motion_masks_status.json','motion_masks_dispatch.json','native_scene_dispatch.json','native_scene_priority_dispatch.json','native_scene_mask_priority_dispatch.json','motion_suite_dispatch.json','native_actor_source_audit.json','motion_suite_engineering_ledger.json','quota_state.json','monitor_state.json','phase2_git_publish.json','benchmark_expansion_dispatch.json','benchmark_masks_dispatch.json','benchmark_ledger_dispatch.json','native_scene4_dispatch.json','native_scene5_dispatch.json','sh_rotation_cpu_control.json','sh_rotation_GPU_dispatch.json','continuous_backend_first_dispatch.json']:
    p=ROOT/name
    if p.exists():shutil.copy2(p,DEST/name)
for name in ['native_scene6_heldout_base_dispatch.json','benchmark_second_execution_dispatch.json','benchmark_CPU_delivery_repair_receipt.json','backend_same_input_replay_dispatch.json','native_layer_isolation_dispatch.json','benchmark_mask_development_priority_dispatch.json','native_smoke_cost_ledger.json']:
    if (ROOT/name).exists():shutil.copy2(ROOT/name,DEST/name)
if (ROOT/'checkpoint_storage_relocation_v1/receipt.json').exists():shutil.copy2(ROOT/'checkpoint_storage_relocation_v1/receipt.json',DEST/'checkpoint_storage_relocation.json')
archive=ROOT/('admission_delivery_delta.tar.gz' if args.delta_base else 'admission_delivery_v2.tar.gz')
snapshot={}
for path in sorted(DEST.rglob('*')):
    if path.is_file() and path.name!='delivery_snapshot_manifest.json':
        snapshot[path.relative_to(DEST).as_posix()]=dict(bytes=path.stat().st_size,sha256=digest_file(path))
base=json.loads(args.delta_base.read_text()).get('files',{}) if args.delta_base else {}
changed=[name for name,value in snapshot.items() if base.get(name)!=value]
manifest=dict(files=snapshot,transferred_files=changed,remote_snapshot_absent_local_paths=sorted(set(base)-set(snapshot)),local_deletions_performed=False)
(DEST/'delivery_snapshot_manifest.json').write_text(json.dumps(manifest,indent=2))
temp=archive.with_suffix('.tmp')
with tarfile.open(temp,'w:gz') as tar:
    for name in changed:tar.add(DEST/name,arcname='admission/'+name,recursive=False)
    tar.add(DEST/'delivery_snapshot_manifest.json',arcname='admission/delivery_snapshot_manifest.json',recursive=False)
temp.replace(archive)
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':digest_file(archive),'snapshot_files':len(snapshot),'transferred_files':len(changed),'local_deletions_performed':False}))
