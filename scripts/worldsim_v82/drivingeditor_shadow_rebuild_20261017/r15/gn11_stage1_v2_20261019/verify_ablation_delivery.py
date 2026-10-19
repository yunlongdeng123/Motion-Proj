"""Static links, architecture and immutable deployed arm archive checks."""
from pathlib import Path
from html.parser import HTMLParser
import hashlib,json,xml.etree.ElementTree as ET,zipfile
W=Path(__file__).resolve().parent
O=W.parent.parent/'outputs/gn11_stage1_v2_20261019/ablations'
class Links(HTMLParser):
    def __init__(self):super().__init__();self.refs=[]
    def handle_starttag(self,tag,attrs):self.refs.extend(v for k,v in attrs if k in ['src','href'] and v)
r=json.loads((W/'ablation_runtime.json').read_text(encoding='utf-8'));p=Links();p.feed((O/'index.html').read_text(encoding='utf-8'))
for ref in p.refs:assert (O/ref).is_file(),ref
ET.parse(O/'architecture.svg')
assert hashlib.sha256((O/'source.zip').read_bytes()).hexdigest()==r['deployment']['archive_sha256']
with zipfile.ZipFile(O/'source.zip') as z:
    assert z.testzip() is None
    for arm,a in r['arms'].items():
        manifest=json.loads(z.read(arm+'/candidate_manifest.json'));assert manifest==a['manifest']
        for name,item in manifest['files'].items():
            digest=item['candidate_sha256'] if isinstance(item,dict) else item
            assert hashlib.sha256(z.read(arm+'/candidate/'+name)).hexdigest()==digest
media=json.loads((O.parent/'native30k_media.json').read_text(encoding='utf-8'))
for item in media['videos'].values():assert hashlib.sha256((O.parent/item['path']).read_bytes()).hexdigest()==item['sha256']
output={'status':'PASSED_ABLATION_REPORT_FILES_AND_DEPLOYED_SOURCE','local_link_references':len(p.refs),'architecture_XML_valid':True,
    'deployed_source_ZIP_and_three_manifests_valid':True,'original_eight_video_hashes_valid':True,'browser_UI_tested':False,'quality_accepted':False}
(O/'local_delivery_checks.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
print(json.dumps(output))
