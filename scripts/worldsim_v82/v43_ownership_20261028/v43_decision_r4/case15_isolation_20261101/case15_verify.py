import pathlib,json,zipfile,collections,ast
from html.parser import HTMLParser
import numpy as np
from PIL import Image
W=pathlib.Path(__file__).resolve().parent;H=W/'case15';O=W.parent/'outputs'/'机制隔离15例_20261101';read=lambda p:json.loads(p.read_text(encoding='utf8'))
d=read(O/'核查与待办.json');assert len(d['rows'])==15 and len({r['case'] for r in d['rows']})==15
ui=read(O/'3D视图代码核验.json');assert len(ui['passed'])==8 and not ui['source_data_modified']
assert sum(d['summary']['research_sample_roles'].values())==15
assert all(len(r['measured_LiDAR']['measured_frame_counts'])==49 for r in d['rows'])
assert all((O/r['measured_LiDAR']['plot']).is_file() for r in d['rows'])
assert all([m['number'] for m in r['mechanisms']]==list(range(1,14)) for r in d['rows'])
assert sum(len(r['source']['raw_window'])==2 for r in d['rows'])==15
assert sum(r['source']['source_pose_frames'] for r in d['rows'])==735
assert all(r['source']['source_track_gap_count']==0 and r['source']['timestamps_strictly_increasing'] and r['source']['box_records_finite_positive'] for r in d['rows'])
assert sum(len(r['geometry']['frames']) for r in d['rows'] if r['geometry'])==588
assert all(r['geometry']['all_export_fields_finite'] for r in d['rows'] if r['geometry'])
assert all(set(r['formal_operations'].values())=={'unknown'} for r in d['rows'])
counts={};files=0
for key in ['waymo-train-212_a35','waymo-train-212_a56','waymo-train-564_a22']:
 m=read(H/'prepared_source_masks'/key/'manifest.json');assert len(m['frames'])==49 and m['nonzero_frames']==49 and not m['independent_pixel_GT'] and not m['used_to_update_G_or_train'] and m['human_verdict'] is None
 for r in m['frames']:
  p=H/'prepared_source_masks'/key/'body_masks'/(r['frame_id']+'.png')
  with Image.open(p) as im:assert im.mode=='L' and im.size==(1920,1280);assert int((np.asarray(im)>0).sum())==r['pixels']
  files+=1
 counts[key]=49
zipout=O/'候选源掩码_3例147帧.zip'
with zipfile.ZipFile(zipout,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted((H/'prepared_source_masks').rglob('*')):
  if p.is_file():z.write(p,p.relative_to(H/'prepared_source_masks'))
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.ids=[]
 def handle_starttag(self,t,a):
  v=dict(a)
  if 'id'in v:self.ids.append(v['id'])
  if v.get('href'):self.links.append(v['href'])
  if v.get('src'):self.links.append(v['src'])
p=Links();p.feed((O/'逐例机制隔离与待办.html').read_text(encoding='utf8'))
for link in p.links:
 if link.startswith('#'):assert link[1:] in p.ids,link
 elif not link.startswith(('http:','https:')):assert (O/link).is_file(),link
assert len(p.ids)==len(set(p.ids)) and all(r['case'] in p.ids for r in d['rows'])
for path in W.glob('case15_*.py'):ast.parse(path.read_text(encoding='utf8'))
proof={'task_id':d['task_id'],'unique_cases':15,'mechanism_entries':195,'source_pose_frames_checked':735,'full_G_case_frames_checked':588,'full_G_cases':12,'provisional_mask_frames_verified':files,'provisional_masks_by_case':counts,'new_GPU_queries':0,'new_training_updates':0,'reused_native_float_records':24,'HTML_local_links_exist':True,'formal_case_operation_unknown':60,'real_browser_QA':'not passed; known file:// URL policy block; no bypass attempted','python_source_syntax':'passed','semantic_limits':'nonzero masks and source ID consistency do not establish independent pixel GT or formal operation eligibility','remaining':'native contributor/depth/shared-route controls; source masks for2cases; frozenG+mask registration for3 new scenes'}
(O/'交付核验.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(proof,ensure_ascii=False))
proof.update({'measured_LiDAR_cases':15,'measured_LiDAR_case_frames':735,'research_sample_roles':d['summary']['research_sample_roles'],'candidate_page_3D_code_checks':ui['passed'],'candidate_page_source_data_modified':False,'page_patch_zip':str(O/'3D_LiDAR页面更新包.zip')})
(O/'交付核验.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf8')
