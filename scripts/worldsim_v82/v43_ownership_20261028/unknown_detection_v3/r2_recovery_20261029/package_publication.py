"""打包本轮复现源码和小型验收记录；大模型、全浮点保留原现场。"""
from pathlib import Path
import tarfile, json

ROOT=Path(__file__).resolve().parents[2]
W=Path(__file__).resolve().parent
REPORT=ROOT/'outputs/V43_Ownership_20261027/gpu_v43/r2_recovery_20261029'
CODE=Path('scripts/worldsim_v82/v43_ownership_20261028')
DEST=CODE/'unknown_detection_v3/r2_recovery_20261029'
DOC=Path('docs/autoresearch/v43_ownership_20261028/r2_recovery_20261029')
items={}
for name in ['r2_inherited_runner.py','scope.py','audit_r2_delivery.py','r2_quality_stats.py','build_review.py','patch_ui.py','install_r2_inherited_ui.py','qa_r2_inherited_ui.js','qa_ui.js','package_publication.py','publish_recovery.py']:
    items[DEST/name]=W/name
for name in ['build_scene_workspaces.py','scene_target_detection.js']:
    items[CODE/'diagnostic_template'/name]=ROOT/'work/persistent_world_20261027'/name
items[CODE/'unknown_detection_v3/scene_target_detection.js']=ROOT/'work/v43_unknown_detection_20261028/scene_target_detection.js'
items[CODE/'unknown_detection_v3/temporal_multiscene_20261029/install_temporal_ui.py']=ROOT/'work/v43_temporal_multiscene_20261029/install_temporal_ui.py'
for p in REPORT.iterdir():
    if p.is_file():items[DOC/p.name]=p
for name in ['algorithm_audit.json','ai_visual_review.json','ui_repair_audit.json','r2_inherited_canvas_qa.json','r2_quality_stats.json']:
    items[DOC/name]=W/name
for name in ['local_audit.json','anchor_replay_validation.json','contact_anchor_validation.json','seed_anchor_validation.json','seed_full49_validation.json','extend_anchor_validation.json']:
    items[DOC/name]=W/'r2_delivery'/name
items[DOC/'run_summary.json']=W/'r2_delivery/r2_inherited49/summary.json'
assert all(p.is_file() for p in items.values())
with tarfile.open(W/'publication.tar.gz','w:gz') as ar:
    for dst,src in items.items():ar.add(src,arcname=str(dst).replace('\\','/'))
(W/'publication_manifest.json').write_text(json.dumps({str(k):str(v) for k,v in items.items()},ensure_ascii=False,indent=2),encoding='utf8')
print('packaged',len(items),'files', (W/'publication.tar.gz').stat().st_size,'bytes')
