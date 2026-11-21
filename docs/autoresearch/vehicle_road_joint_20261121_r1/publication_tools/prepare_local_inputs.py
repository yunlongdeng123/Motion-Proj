"""按最终授权正清单打包本地报告源；不暂存或提交 Git。"""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile
from archive_common import ARCHIVE, lightweight_text

WORKSPACE = Path(__file__).resolve().parents[3]
W = WORKSPACE/'work/vehicle_road_joint_20261121'
O = WORKSPACE/'outputs/vehicle_road_joint_20261121'
PUB = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main(output, remote_root):
    output=Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    entries=[]
    def add(relative, source, role):
        data=lightweight_text(source)
        target=output/'tree'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        entries.append(dict(source=remote_root+'/tree/'+relative,destination=ARCHIVE+'/'+relative,
                            role=role,sha256=sha(data),bytes=len(data),
                            local_origin=dict(path=str(Path(source).resolve()),sha256=sha(data))))
    for name in ['build_report.py','assemble_report_data.py','report_template.html','viewer.js',
                 'link_existing_report.py','qa_report.py','qa_extras.py','independent_detail_bench.py',
                 'joint_detail_metrics.py','road_detail_metrics.py','qa_a20_visibility.py']:
        add('report_source/'+name,W/name,'report_source')
    add('report_source/report.md',W/'report.md','report_source')
    add('report_source/generated_report.md',O/'report.md','report_source')
    for name in ['index.html','report_data.json']:
        add(name,O/name,'report_source')
    for name in ['hugsim_review.md','a20_occlusion_qa.md','final_independent_qa.md']:
        add(name,W/name,'record')
    for f in sorted(W.glob('qa_*.json')):
        add('evidence/local_qa/'+f.name,f,'receipt')
    for name in ['actor_mapping.json','takeback_download_receipt.json','implementation_delivery_receipt.json','report_entry_receipt.json']:
        add('local_receipts/'+name,W/name,'receipt')
    add('architecture.svg',O/'architecture.svg','architecture_source')
    for name in ['generate.py','render.ps1','render.html','content.json']:
        add('report_source/architecture/'+name,W/'architecture'/name,'architecture_source')
    add('linked_sources/road_surface_residual_20261121_r2/viewer.js',WORKSPACE/'work/road_surface_residual_20261121_r2/viewer.js','provenance')
    for f in sorted(PUB.glob('*.py')):
        add('publication_tools/'+f.name,f,'publication_tool')
    for name in ['README.md','manifest.template.json']:
        add('publication_tools/'+name,PUB/name,'publication_tool')
    # 归档主报告只转换链接；原始本地文本另外逐字保存。
    original=(W/'report.md').read_text(encoding='utf-8')
    portable=original.replace('C:/Users/dengyunlong/Documents/Codex/2026-11-16/xian/outputs/vehicle_road_joint_20261121/architecture.png','architecture.svg')
    portable=portable.replace('C:/Users/dengyunlong/Downloads/HUGSIM.pdf','https://arxiv.org/abs/2412.01718')
    pp=output/'derived/report.md';pp.parent.mkdir(parents=True,exist_ok=True);pp.write_text(portable,encoding='utf-8')
    add('report.md',pp,'report_source')
    bindings={r['destination']:r['local_origin'] for r in entries if not r['destination'].endswith('/report.md') or '/report_source/' in r['destination']}
    provenance=output/'derived/local_delivery_source_bindings.json'
    provenance.write_text(json.dumps(dict(archive_link_conversion_only=True,raw_local_report_preserved=True,
                          files=bindings),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    add('provenance/local_delivery_source_bindings.json',provenance,'provenance')
    catalog=dict(publication_ready=True,root_report_complete=True,formal_stage=False,
                 sources=entries,total_bytes=sum(r['bytes'] for r in entries))
    (output/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    zpath=output.with_suffix('.zip')
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
        z.write(output/'catalog.json','catalog.json')
        for f in (output/'tree').rglob('*'):
            if f.is_file():z.write(f,f.relative_to(output).as_posix())
    (output/'upload_receipt.json').write_text(json.dumps(dict(path=str(zpath),sha256=sha(zpath.read_bytes()),bytes=zpath.stat().st_size),indent=2)+'\n')
    print(json.dumps(dict(zip=str(zpath),files=len(entries),bytes=catalog['total_bytes'],formal_stage=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--remote-root',required=True)
    a=p.parse_args();main(a.output,a.remote_root)
