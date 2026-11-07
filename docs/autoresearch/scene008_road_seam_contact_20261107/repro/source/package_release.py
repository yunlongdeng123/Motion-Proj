"""把已完成的科学结果与重放状态收集为离线交付；不加载模型。"""
from pathlib import Path
import hashlib
import json
import shutil
import tarfile

RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_road_seam_contact_20261107_r1')
OUT=RUN/'delivery_release_v2'
assert (OUT/'index.html').exists()
assert not (OUT/'publication_manifest.json').exists()
added={}
def copy(src,rel):
    src=Path(src);dest=OUT/rel;dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        assert dest.read_bytes()==src.read_bytes(),rel
    else:shutil.copy2(src,dest)
    added[rel]=str(src)

for src in (RUN/'contract').glob('*.json'):copy(src,'repro/contracts/'+src.name)
copy(RUN/'round_decision.json','repro/round_decision.json')
copy(RUN/'build_report.py','repro/build_report.py')
for src in (RUN/'source').glob('*.py'):copy(src,'repro/source/'+src.name)
for src in (RUN/'engineer').glob('*'):
    if src.is_file() and src.suffix in ('.py','.json'):copy(src,'repro/engineer/'+src.name)
canonical=RUN/'engineer/canonical_n2_mip_candidate_v2'
for src in canonical.rglob('*'):
    if src.is_file() and src.suffix in ('.py','.json'):
        copy(src,'repro/canonical/'+src.relative_to(canonical).as_posix())
for folder in ['n1_parity','n2_pilot','n2_full49','n2_mip_pilot','n2_mip_full49','n2_material_pilot','n2_material_full49','n2_mip_material_pilot']:
    for src in (RUN/'results'/folder/'source_snapshot').glob('*'):
        if src.is_file():copy(src,'repro/executed_source/'+folder+'/'+src.name)
    copy(RUN/'results'/folder/'receipt.json','repro/result_receipts/'+folder+'.json')
copy(RUN/'diagnostic/layers_v1/receipt.json','repro/result_receipts/layers_v1.json')
# 保留材质对照的完整法方程/恢复/求解状态；它不属于主资产。
fitroot=RUN/'blue/boundary_material_v1'
for src in fitroot.rglob('*'):
    if src.is_file() and src.suffix in ('.npz','.json','.py'):
        assert src.stat().st_size<10_000_000
        copy(src,'repro/boundary_material/'+src.relative_to(fitroot).as_posix())
copy(RUN/'blue/boundary_material_control.py','repro/boundary_material/boundary_material_control.py')
for src in (RUN/'blue').glob('*'):
    if src.is_file() and src.suffix in ('.json','.md','.py'):
        copy(src,'repro/blue/'+src.name)
for src in (RUN/'red').glob('*'):
    if src.is_file() and src.suffix in ('.json','.md','.py','.npz'):
        assert src.stat().st_size<10_000_000
        copy(src,'repro/red/'+src.name)

state_archive=OUT/'downloads/boundary_material_replay_states.tar.gz'
with tarfile.open(state_archive,'w:gz') as tar:
    tar.add(OUT/'repro/boundary_material',arcname='boundary_material',recursive=True)
html=(OUT/'index.html').read_text()
extra='''<section id="complete-replay"><h2>已保存的重放与对照状态</h2><p>主资产只采用n2＋AA及父cellfit1000材质。下列18系数材质法方程、真实恢复与求解状态仅用于复现未选入的颜色对照。</p><p><a href="downloads/boundary_material_replay_states.tar.gz">材质对照完整重放状态</a> · <a href="repro/canonical/scene_manifest.json">主道路场景manifest</a> · <a href="repro/canonical/final_review_v2.json">带状态/边界/分母的独立最终review</a> · <a href="repro/round_decision.json">本轮分项结论与计算总账</a> · <a href="repro/source/evaluate_refinement.py">实际评价入口</a></p><p>原V44模型、完整传感输入与大浮点现场仍保留在AutoDL；轻量包不代表原始数据和全部检查点备份。manifest中导出时的NOT_RUN记录保持原样，后续独立10次重放的通过证据另存于相同目录的parity收据。</p></section>'''
assert 'id="complete-replay"' not in html
html=html.replace('<footer>',extra+'<footer>',1) if '<footer>' in html else html.replace('<script id="report-data"',extra+'<script id="report-data"',1)
(OUT/'index.html').write_text(html)
(OUT/'publication_additions.json').write_text(json.dumps(dict(extra_copies=added,scientific_sources_changed=False,main_candidate='n2+AA; parent colours; blue control not selected',prior_build_manifest='file_manifest.json'),ensure_ascii=False,indent=2)+'\n')
manifest={}
for src in OUT.rglob('*'):
    if src.is_file():manifest[src.relative_to(OUT).as_posix()]={'bytes':src.stat().st_size,'sha256':hashlib.sha256(src.read_bytes()).hexdigest()}
(OUT/'publication_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
archive=RUN/'delivery_release_v2.tar.gz'
assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:tar.add(OUT,arcname=OUT.name)
receipt=dict(archive_prefix=OUT.name,bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    files=len(manifest)+1,primary='n2_mip_full49',main_candidate_material='parent cellfit1000',
    whole_scene_adopted=False,AutoDL='KEEP_ON',source=str(OUT))
(RUN/'delivery_packet_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
