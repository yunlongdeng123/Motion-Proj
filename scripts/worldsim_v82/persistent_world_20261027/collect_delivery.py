"""运行中里程碑收集；不停止控制器，不把快照当作研究完成。"""
import json,subprocess,tarfile,time,shutil
from pathlib import Path
from world_state import WorldState
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')
native=ROOT/'phase1_native';ops=ROOT/'phase2_operations'
state=WorldState(native/'world.sqlite','waymo_scene0008_development')
for name in ['delete_selected','move_selected','insert_reused','camera_shift','camera_revisit']:
    state.revision(name,'background_preserve_199',dict(operation=name,asset_binding=str(ops/'dynamic_binding.npz'),
        source='原生4D高斯工程基线',semantic_binding_verified=False))
    state.record_request(name,name,dict(operation=name,frames=list(range(89,138))),ops/name)
for case in ['delete_all_unrepaired','delete_all_repaired','delete_all_evidence_routed']:
    path=native/case/'propainter/summary.json'
    if path.exists():
        payload=dict(outputs=str(path.parent/'model_png'),mask=str(native/case/'mask'),
            native_state='native' if case=='delete_all_unrepaired' else 'background_preserve_199',
            promoted_to_observed=False,promoted_to_geometry=False)
        with state.db:state.db.execute('INSERT OR IGNORE INTO hypothesis VALUES(?,?,?,?,?)',
            (state.scene,case,payload['native_state'],'ProPainter',json.dumps(payload)))
summary=state.summary();state.close()
snap=dict(persistent_state=summary,snapshot_time=time.time(),research_complete=False,quality_accepted=False,human_verdict=None)
for name in ['controller_status','training_status','gpu_peak','download_status']:
    p=ROOT/(name+'.json')
    if p.exists():snap[name]=json.loads(p.read_text())
snap['reference49_completed']=len(list((ROOT/'phase2_reference49/crossview').glob('*.npz')))
(ROOT/'milestone_snapshot.json').write_text(json.dumps(snap,indent=2))
videos=[]
ffmpeg=shutil.which('ffmpeg')
if not ffmpeg:
    candidates=list(Path('/root/autodl-tmp/envs').glob('*/lib/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*'))
    ffmpeg=str(candidates[0]) if candidates else None
for parent in [native,ops]:
    for case in parent.iterdir():
        if not case.is_dir() or not (case/'native_rgb').exists():continue
        for sub in ['native_rgb','propainter/model_png']:
            folder=case/sub
            if not folder.exists():continue
            frames=sorted(folder.glob('*.png'))
            if len(frames)!=49:continue
            output=case/('native_h264.mp4' if sub=='native_rgb' else 'generated_h264.mp4')
            if not output.exists() and ffmpeg:
                subprocess.run([ffmpeg,'-v','error','-y','-framerate','10','-start_number','89','-i',str(folder/'%03d.png'),
                    '-frames:v','49','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',str(output)],check=True)
            if output.exists():videos.append(str(output.relative_to(ROOT)))
snap['videos']=videos
(ROOT/'milestone_snapshot.json').write_text(json.dumps(snap,indent=2))
archive=ROOT/'milestone_delivery.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in ROOT.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(ROOT)
        if 'code' in rel.parts and p.suffix=='.py':tar.add(p,arcname=str(rel))
        elif rel.parts[0] in ['phase1_native','phase2_operations'] and p.suffix in ['.png','.mp4','.json','.sqlite']:
            tar.add(p,arcname=str(rel),recursive=False)
        elif len(rel.parts)==1 and p.suffix=='.json':tar.add(p,arcname=str(rel))
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,videos=len(videos),snapshot=snap),indent=2))
