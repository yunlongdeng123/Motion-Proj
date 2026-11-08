"""本地资源、构建/留出隔离、实际CPU回执与控制入口的验收。"""
from pathlib import Path
from html.parser import HTMLParser
import ast
import hashlib
import json
import subprocess
from PIL import Image

BASE=Path(__file__).resolve().parents[2]
OUT=BASE/'outputs/waymo_val065_preparation'
HERE=Path(__file__).resolve().parent


class Page(HTMLParser):
    def __init__(self):
        super().__init__();self.ids=[];self.resources=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        for k in ['src','href']:
            if a.get(k) and not a[k].startswith(('#','http:','https:')):self.resources.append(a[k])


def main():
    read=lambda p:json.loads(p.read_text(encoding='utf8'))
    contract=read(OUT/'experiment_contract.json')
    receipt=read(OUT/'evidence/inputs/receipt.json')
    verification=read(OUT/'evidence/inputs/verification.json')
    point=read(OUT/'evidence/inputs/point_prompt_verification.json')
    assert receipt['status']=='PASS_CPU_OBSERVATIONS' and verification['status']=='PASS_CPU_BANK'
    assert len(contract['build'])==159 and contract['holdout']==list(range(79,119))
    assert not set(contract['build'])&set(contract['holdout'])
    actual={int(p.stem) for p in (OUT/'assets/input').glob('*.jpg')}
    assert actual==set(contract['build'])
    data=json.loads((OUT/'assets/source.js').read_text(encoding='utf8').removeprefix('window.CPU_CASE=').removesuffix(';'))
    for row in data['frames']:
        p=OUT/row['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
        with Image.open(p) as im:assert im.size==(960,640);im.verify()
    assert {r['raw'] for r in point['frames']}==set(contract['build'])
    for p in (HERE/'source').glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
    # 无编辑独立脚本没有编辑包导入，也没有编辑API调用；运行验收留给GPU。
    tree=ast.parse((HERE/'source/render_v1.py').read_text(encoding='utf8'))
    imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert not any('drivingeditor_dynamic_edit' in n for n in imports)
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='render_scene' for n in ast.walk(tree))
    page=Page();text=(OUT/'index.html').read_text(encoding='utf8');page.feed(text)
    assert '@@' not in text and len(page.ids)==len(set(page.ids))
    assert all((OUT/r).exists() for r in page.resources),page.resources
    subprocess.run(['node','--check',str(OUT/'assets/viewer.js')],check=True)
    subprocess.run(['node',str(HERE/'test_source_viewer.cjs')],check=True)
    report=dict(status='PASS_CPU_PREPARATION_DELIVERY',build_frames=159,heldout_frames=40,
        original_measured_points=receipt['measured_points'],input_files_CPU_SHA_checked=verification['hashes_checked'],
        locally_copied_observation_frames=159,target_positive_frames=point['target_positive_frames'],
        raw_source_sha256=receipt['raw_sha256'],native_no_edit_entry_AST_checked=True,
        CUDA_executed=False,training_updates=0,render_calls=0,browser_QA='not performed; Node DOM/Canvas logic only',
        scientific_acceptance='CPU inputs only; segmentation, reconstruction and editing require GPU',human_verdict=None)
    (OUT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
