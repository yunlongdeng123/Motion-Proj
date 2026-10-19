"""Check generated local assets and JavaScript syntax; not browser UI testing."""
from pathlib import Path
from html.parser import HTMLParser
import ast,json,re,subprocess,xml.etree.ElementTree as ET,zipfile
from PIL import Image

W=Path(__file__).resolve().parent;O=W.parent.parent/'outputs/gn11_stage1_v2_20261019'
class Links(HTMLParser):
    def __init__(self):super().__init__();self.paths=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ['src','href','data-image','data-query-frame'] and value and not value.startswith(('https:','http:','#')):self.paths.append(value)
h=(O/'index.html').read_text(encoding='utf-8');p=Links();p.feed(h)
for link in p.paths:
    if not (O/link).is_file():raise AssertionError('missing local link '+link)
scripts=re.findall(r'<script>(.*?)</script>',h,re.S)
for i,s in enumerate(scripts):
    path=W/f'report_script_{i}.js';path.write_text(s,encoding='utf-8')
    subprocess.run(['node','--check',str(path)],check=True,capture_output=True)
pngs={O/link for link in p.paths if link.endswith('.png')}
for f in pngs:
    with Image.open(f) as im:assert im.size==(480,320)
ET.parse(O/'architecture.svg')
for f in W.rglob('*.py'):ast.parse(f.read_text(encoding='utf-8'))
with zipfile.ZipFile(O/'candidate_source.zip') as z:
    assert z.testzip() is None
    assert 'candidate/LICENSE_gaussian_splatting.md' in z.namelist()
    for name in ['config.json','cpu_test_receipt.json','candidate_manifest.json']:
        assert z.read(name)==(W/name).read_bytes()
receipt=dict(status='PASSED_LOCAL_FILES_AND_SYNTAX',local_link_references=len(p.paths),
    JS_syntax_checks=len(scripts),fullframe_PNGs=len(pngs),architecture_XML_valid=True,source_AST_valid=True,
    browser_UI_tested=False,actual_remote_inputs_verified=(O/'actual_road_summary.json').exists(),
    GPU_tested=(O/'actual_gpu_gate_receipt.json').exists(),training_started=(W/'runtime_progress.json').exists())
(O/'local_delivery_checks.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt))
