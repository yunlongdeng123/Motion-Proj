from pathlib import Path
from html.parser import HTMLParser
from collections import Counter
import json,hashlib,subprocess,sys
w=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-31/g/work/actor_diagnostics_20261101/ui');p=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-22/ni-x-a/outputs/GS_Scene_Diagnostics_20261027/actor_candidates');q=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-31/g/outputs/V43_R4_GPU_20261031/actor_candidates');out=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-31/g/outputs/Actor_Diagnostics_20261101/evidence/ui_regression_receipt.json')
class H(HTMLParser):
 def __init__(self):super().__init__();self.ids=[];self.refs=[];self.links=[]
 def handle_starttag(self,t,a):
  d=dict(a)
  if d.get('id'):self.ids.append(d['id'])
  if t=='script' and d.get('src'):self.refs.append(d['src'])
  if t=='link' and d.get('href'):self.refs.append(d['href'])
  if t=='a' and d.get('href'):self.links.append(d['href'])
tests=['test_sync.js','test_hash.js','test_trajectory_full.js','test_212_gs_media.js','test_537_real.js','test_frame_failure.js','test_zero_actor.js'];all_tests=[];errors=[];pages=[]
for label,root in [('shared',p),('mirror',q)]:
 h=H();h.feed((root/'index.html').read_text(encoding='utf8'));bad_ids=[k for k,v in Counter(h.ids).items() if v>1];missing_refs=[r for r in h.refs if not (root/r.split('?',1)[0]).resolve().is_file()];css=(root/'sync_panel.css').read_text(encoding='utf8');oldcss=(root/'index.html').read_text(encoding='utf8');scripts=['data.js','actor_names.js','actor_annotations.js','actor_eligibility.js','viewer.js','sync_panel.js','geometry/gs_evidence.js'];syntax={}
 for f in scripts:
  x=subprocess.run(['node','--check',str(root/f)],capture_output=True,text=True,encoding="utf-8",errors="replace");syntax[f]=x.returncode==0
  if x.returncode:errors.append(label+' syntax '+f+' '+x.stderr)
 for f in tests:
  x=subprocess.run(['node',str(w/f),str(root)],capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=40);all_tests.append({'page':label,'test':f,'passed':x.returncode==0,'output':x.stdout.strip(),'error':x.stderr.strip()})
  if x.returncode:errors.append(label+' '+f+' '+x.stderr)
 info={'page':label,'path':str(root),'duplicate_ids':bad_ids,'missing_static_refs':missing_refs,'syntax':syntax,'overlay_canvas_transparent':'background:transparent' in css and 'pointer-events:none;background:transparent' in oldcss,'report_links':[v for v in h.links if 'report' in v.lower() or 'status' in v.lower()]}
 if bad_ids or missing_refs or not info['overlay_canvas_transparent'] or info['report_links']:errors.append(label+' structural checks failed')
 pages.append(info)
root=(p.parent/'index.html').read_text(encoding='utf8');root_counts={'scenario_card_count':root.count('class="primary"'),'actor_candidates_link_count':root.count('href="actor_candidates/index.html"')}
if root_counts!={'scenario_card_count':4,'actor_candidates_link_count':1}:errors.append('root navigation changed unexpectedly '+str(root_counts))
parity={f:hashlib.sha256((p/f).read_bytes()).hexdigest()==hashlib.sha256((q/f).read_bytes()).hexdigest() for f in ['data.js','actor_names.js','actor_annotations.js','actor_eligibility.js','viewer.js','sync_panel.js','sync_panel.css','geometry/gs_evidence.js']}
if not all(parity.values()):errors.append('shared/mirror source mismatch')
source_sha256={label:{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ['index.html','data.js','actor_names.js','actor_annotations.js','actor_eligibility.js','viewer.js','sync_panel.js','sync_panel.css','geometry/gs_evidence.js']} for label,root in [('shared',p),('mirror',q)]}
result={'schema':'actor_diagnostics_ui_regression_v1','passed':not errors,'source_sha256':source_sha256,'pages':pages,'root_navigation':root_counts,'shared_mirror_source_parity':parity,'tests':all_tests,'errors':errors,'limitations':['Node tests simulate DOM and field loading; no browser interaction is claimed.','frame/image/LiDAR comparisons use actual source data where indicated in test output; gap and zero cases remain source-bound.','File URL browser restrictions were not bypassed.','The 212 cross-runtime strict 2e-5 coordinate check separately failed (approximately 0.976 mm); these UI tests do not assert geometric parity or formal eligibility.']}
if errors:
 print(json.dumps(result,ensure_ascii=False,indent=2),file=sys.stderr);sys.exit(1)
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print('PASS',len(all_tests),'DOM checks and static checks; receipt',out)
