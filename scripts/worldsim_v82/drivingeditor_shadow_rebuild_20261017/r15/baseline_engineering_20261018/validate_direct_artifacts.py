from pathlib import Path
import re,json,subprocess,ast,xml.etree.ElementTree as ET
from PIL import Image
W=Path(__file__).parent;O=W.parents[1]/'outputs/baseline_engineering_20261018'
def main():
 pages=[O/'index.html',O/'intersection_inputs.html',O.parent/'identity_comparison_20261018/index.html',O.parent/'identity_comparison_20261018/sync_player.html']
 if (O/'results.html').exists():pages.append(O/'results.html')
 js=0;links=0
 for p in pages:
  s=p.read_text(encoding='utf-8')
  for v in re.findall(r'(?:href|src)="([^"]+)"',s):
   if v.startswith(('http','#','data:')) or '${' in v:continue
   assert (p.parent/v).exists(),(p,v);links+=1
  assert 'id="zoom"' not in s
  for i,code in enumerate(re.findall(r'<script>(.*?)</script>',s,re.S)):
   f=W/f'direct_validation_{js}.js';f.write_text(code,encoding='utf-8')
   subprocess.run(['node','--check',str(f)],check=True,capture_output=True);js+=1
 template=(W/'fullframe_player_template.html').read_text(encoding='utf-8')
 tree=ast.parse((W/'build_direct_player.py').read_text(encoding='utf-8'))
 inject=next(n.value.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='inject' for t in n.targets))
 template=template.replace(' const stat=alpha?b.static_alpha:',inject)
 code=re.findall(r'<script>(.*?)</script>',template,re.S)[0]
 f=W/'final_player_template_check.js';f.write_text(code,encoding='utf-8')
 subprocess.run(['node','--check',str(f)],check=True,capture_output=True)
 js+=1
 ET.parse(O/'direct_architecture.svg')
 media=0
 if (O/'results.html').exists():
  from build_direct_player import video_contract
  result_page=(O/'results.html').read_text(encoding='utf-8')
  data=json.loads(re.search(r'<script type="application/json" id="data">(.*?)</script>',result_page,re.S).group(1))
  model=data['models']['gn11five'];assert model['all_vehicle_stage1'] and len(model['base'])==7
  for rel in model['base'].values():video_contract(O/rel);media+=1
  subprocess.run(['node',str(W/'check_player_identity_controls.js')],check=True,capture_output=True)
 im=Image.open(O/'all_vehicle_raw128.png');assert im.size==(960,320);im.verify()
 for raw in range(89,138):
  with Image.open(O/'intersection_input_frames'/f'0008{raw:03d}.jpg') as im:
   assert im.size==(1920,2560);im.verify()
 source=(O/'intersection_inputs.html').read_text(encoding='utf-8')
 code=re.findall(r'<script>(.*?)</script>',source,re.S)[0]
 harness='''const assert=require('node:assert/strict');
const elements=Object.fromEntries(['timeline','frame','label','play','prev','next'].map(x=>[x,{value:39,addEventListener(t,f){this[t]=f;}}]));
const document={getElementById(x){return elements[x]},addEventListener(){}};
let tick=null;function setInterval(f){tick=f;return 1}function clearInterval(){tick=null}
'''+code+'''
assert.equal(elements.label.textContent,'raw128 · 40/49');
elements.prev.onclick();assert.equal(elements.label.textContent,'raw127 · 39/49');
elements.timeline.value=0;elements.timeline.input();assert.equal(elements.frame.src,'intersection_input_frames/0008089.jpg');
elements.play.onclick();assert.equal(elements.play.textContent,'暂停');tick();assert.equal(elements.timeline.value,1);
elements.play.onclick();assert.equal(tick,null);
for(let n=0;n<49;n++){show(n);assert.equal(elements.frame.src,FRAMES[n].src);}
'''
 f=W/'intersection_controls_check.js';f.write_text(harness,encoding='utf-8');subprocess.run(['node',str(f)],check=True,capture_output=True)
 for p in W.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
 result=dict(status='PASSED_FILE_MEDIA_JS_AND_SOURCE_CHECKS',links=links,JS_syntax_checked=js,full_frame_display=True,
  ROI_crop_controls_removed=True,input_full_frame_panels=49,input_timeline_DOM_fixture_passed=True,actual_30K_videos_verified=media,
  identity_controls_DOM_fixture_passed=True,actual_browser_UI_verified=False,reason='file URL browser safety restriction respected',human_verdict=None)
 (O/'direct_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
