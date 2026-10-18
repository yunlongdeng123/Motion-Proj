from pathlib import Path
import re,json,subprocess,xml.etree.ElementTree as ET
from PIL import Image
W=Path(__file__).parent;O=W.parents[1]/'outputs/baseline_engineering_20261018/road_observability_20261019'
page=(O/'index.html').read_text(encoding='utf-8');links=0
for rel in re.findall(r'(?:src|href)="([^"]+)"',page):
 if rel.startswith(('http','#')):continue
 assert (O/rel).exists(),rel;links+=1
images=0
for p in (O/'frames').glob('*'):
 with Image.open(p) as im:assert im.size in [(480,320),(960,640)],(p,im.size);im.verify();images+=1
assert images==20,images
ET.parse(O/'architecture.svg')
js=re.search(r'<script>(.*?)</script>',page,re.S).group(1);code=W/'road_observability_controls.js';code.write_text(js,encoding='utf-8')
subprocess.run(['node','--check',str(code)],capture_output=True,check=True)
fixture="const assert=require('node:assert/strict');const el=Object.fromEntries(['raw','gt','static','dynamic','exposure'].map(k=>[k,{value:'128'}]));const document={getElementById:k=>el[k]};"+js+";for(const raw of [89,107,128,137]){el.raw.value=String(raw);el.raw.onchange();for(const key of ['gt','static','dynamic','exposure'])assert.equal(el[key].src,'frames/raw'+raw+'_'+key+'.png');}"
code=W/'road_observability_fixture.js';code.write_text(fixture,encoding='utf-8');subprocess.run(['node',str(code)],capture_output=True,check=True)
d=json.loads((O/'evidence_summary.json').read_text(encoding='utf-8'));assert d['frames']==49 and d['silhouette_source']['diagnostic_source_frames']==199 and d['training_updates']==0 and not d['inputs_modified']
for r in d['silhouette_rows']:
 # Native-protobuf and converted-input projection paths yield matching exposure statistics in this window.
 assert abs(r['exposure_49']['never_exposed_fraction']-r['box_only_training49_opportunities']['never_exposed_fraction'])<1e-12
 assert r['exposure_49_mask_dilated_2px']['never_exposed_fraction']>=r['exposure_49']['never_exposed_fraction']
result=dict(status='PASSED_FILES_NUMERIC_CONTRACT_FULLFRAME_IMAGES_AND_JS_FIXTURE',full_frame_images_verified=images,links_checked=links,raw_frame_switch_fixture=True,independent_native_source_projection_agrees=True,training_updates=0,GPU_used=False,browser_interaction_verified=False,human_verdict=None)
(O/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
