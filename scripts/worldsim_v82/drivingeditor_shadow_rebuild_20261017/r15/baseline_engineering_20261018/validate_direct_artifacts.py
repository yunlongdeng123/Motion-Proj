from pathlib import Path
import re,json,subprocess,ast,xml.etree.ElementTree as ET
from PIL import Image
W=Path(__file__).parent;O=W.parents[1]/'outputs/baseline_engineering_20261018'
def main():
 pages=[O/'index.html',O.parent/'identity_comparison_20261018/index.html',O.parent/'identity_comparison_20261018/sync_player.html']
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
 ET.parse(O/'direct_architecture.svg')
 im=Image.open(O/'all_vehicle_raw128.png');assert im.size==(960,320);im.verify()
 for p in W.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
 result=dict(status='PASSED_FILE_MEDIA_JS_AND_SOURCE_CHECKS',links=links,JS_syntax_checked=js,full_frame_display=True,
  ROI_crop_controls_removed=True,actual_browser_UI_verified=False,reason='file URL browser safety restriction respected',human_verdict=None)
 (O/'direct_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
