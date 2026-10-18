"""只打包既有真实PNG与metadata，不重新渲染或下载模型/FP32。"""
import json
from pathlib import Path
import tarfile

BASE=Path('/root/autodl-tmp/drivingeditor_repro/campaign/instance_editor_20261013')
POST=BASE/'post30k_r3'
OUT=BASE/'full30k_export_r3'
def read(path):return json.loads(path.read_text(encoding='utf-8'))

status=read(POST/'status.json')
if status.get('state')!='POST30K_EVALUATION_SIX_EDITS_CPU_CONTRACT_PASS_VISUAL_REVIEW_PENDING':
    raise RuntimeError('必须真实完成评价、六编辑与CPU合同，不伪造full资产')
evaluation=read(POST/'evaluation/scene_evaluation_manifest.json')
summary=read(POST/'edit_summary.json')
assert evaluation['contracts']['observed_frames']==49 and evaluation['contracts']['fixed_camera_frames']==49
assert evaluation['contracts']['model_and_optimizer_unmodified_after_evaluation'] is True
assert summary['engineering_contract']['passed'] is True and summary['engineering_contract']['verified_FP32_pairs']==294
pngs=list(POST.rglob('*.png'))
if len(pngs)!=3969:raise RuntimeError('真实完整PNG数应为3969')
metadata=list(POST.rglob('*.json'))
if OUT.exists():raise FileExistsError('保留旧资产，不覆盖导出包')
OUT.mkdir()
with tarfile.open(OUT/'actual_full30k_PNG_metadata.tar.gz','w:gz') as archive:
    for path in sorted(pngs+metadata):archive.add(path,arcname=path.relative_to(POST).as_posix())
receipt={'status':'ACTUAL_FULL30K_PNG_METADATA_PACKAGED','evaluation_FP32_frames_retained_remotely':98,
         'edit_FP32_pairs_retained_remotely':294,'PNG_count':len(pngs),'metadata_count':len(metadata),
         'source':str(POST),'package':str(OUT/'actual_full30k_PNG_metadata.tar.gz'),
         'bytes':(OUT/'actual_full30k_PNG_metadata.tar.gz').stat().st_size,
         'GPU_used':False,'human_quality_verdict':None}
(OUT/'package_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt))
