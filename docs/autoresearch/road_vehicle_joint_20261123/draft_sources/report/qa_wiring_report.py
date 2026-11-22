from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

WS=Path(r'C:\Users\dengyunlong\Documents\Codex\2026-11-22\065-file-c-users-dengyunlong-documents-2')
OUT=WS/'outputs/road_vehicle_joint_dggt_20261123_r8';QA=OUT/'report_qa';QA.mkdir(parents=True,exist_ok=True)
HTML=Path(r'C:\Users\dengyunlong\Documents\Codex\2026-11-16\xian\outputs\road_material_three_rounds_20261121\index.html')
A20=HTML.parent.parent/'a20_joint_reconstruction_20261122/index.html'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
D=read(OUT/'report_data.json');pub=read(OUT/'report_publication_receipt.json')
BEGIN=b'<!-- ROAD_VEHICLE_JOINT_DGGT_R8_UI_BEGIN -->';END=b'<!-- ROAD_VEHICLE_JOINT_DGGT_R8_UI_END -->'
def outside(data):
 if BEGIN not in data:return data
 start=data.index(BEGIN);end=data.index(END,start)+len(END);return data[:start]+data[end:]
assert outside(HTML.read_bytes())==outside(Path(pub['backup']).read_bytes())
assert sha(A20)=='175c3bd03a01a531ebfcb3e43047949d43952eff9971f46b14a0d19ded415cb2'
assert D['model_result'] is None and D['optimization_updates']==0
assert D['parameter_contract']['centres_frozen'] and D['parameter_contract']['rigid_trajectory_frozen'] and D['parameter_contract']['optimization_currently_paused']
assert list(map(int,D['images']))==[86,110,134,179,198]
B=D['native_baseline_result'];assert B['frames']==[86,110,134,179,198]
assert B['receipt']['actual_nativeF']==15 and B['receipt']['actual_nativeB']==0 and B['receipt']['optimizer_updates']==0 and B['cumulative_nativeF']==25
assert D['rim']['fixed_PNG']==1792 and D['rim']['fixed_float']==1804
assert D['r9']['old_Brown_dark_pixels']==4173 and D['r9']['Gaussian_result'] is None
assert D['r9']['coordinate_status']=='PASS' and D['r9']['surface_status']=='shared_input_PASS' and D['r9']['new_diffusion_calls']==0
C=D['r9']['continuous_v2'];assert C['numeric_status']==C['visual_status']==C['root_status']=='PASS' and C['Gaussian_result'] is None
assert C['field_sha256']=='2eaf1569a97fb8ee6907696dfa28db91bb5f3072f74d37491007c87012e2c647'
assert C['root_file']['sha256']=='fa1490aafd847c4e6bd4e9e2b1193ac6232d6cae29ed8672be4aa4be0652a25f'
assert C['lineage_tail']['kind2_with_nonzero_continuous_teacher_tail_texels']==40850
assert D['r9']['joint_fit']['status']=='not_run' and D['r9']['joint_fit']['result'] is None
S=D['r9']['shared_v1'];assert S['numeric_status']=='PASS' and S['visual_status']=='FAIL' and S['Gaussian_result'] is None
assert S['field_sha256']=='8ad4b9594ef5e14f9f77ebf57a395ebf3da410bb435eb2170f6b2558de9886dd'
assert S['fixed_pair']['fixed_pair_texels']==43624 and S['fixed_pair']['fixed_channel_samples']==130872
assert D['owner']['cumulative_F']==27 and D['owner']['cumulative_B']==2 and D['owner']['updates']==0
assert all(not row['RGBA_changed'] for row in D['owner']['rows'])
assert D['colour_guard']['cumulative_F']==32 and D['colour_guard']['cumulative_B']==7 and D['colour_guard']['updates']==0
assert all(row['source_RGB_max_abs']==row['source_alpha_max_abs']==0 for row in D['colour_guard']['rows'])
assert D['coverage']['new_nativeF']==D['coverage']['new_nativeB']==D['coverage']['optimizer_updates']==0
assert all(row['original_scene_alpha95_low']==row['expanded_scene_alpha95_low']==0 for row in D['coverage']['rows'])
assert [(r['frozen_visible']['isolated_alpha95_low'],r['frozen_visible']['isolated_sufficient_shared_owner_alpha95_low']) for r in D['coverage']['rows'] if r['frozen_visible']]==[(13,5478),(51,6667)]
images=0
for view in B['views'].values():
 for image in view['images'].values():
  full=HTML.parent/image['path'];assert sha(full)==image['sha256']
  roi=image['roi'];x0,y0,x1,y1=roi['bbox']
  assert np.array_equal(np.asarray(Image.open(full))[y0:y1,x0:x1],np.asarray(Image.open(HTML.parent/roi['path'])))
  images+=1
for row in D['r9']['source_images'].values():
 assert row['source_qualification']=='not_evaluated'
 for image in row['images'].values():
  full=HTML.parent/image['path'];assert sha(full)==image['sha256']
  if 'roi' in image:
   roi=image['roi'];x0,y0,x1,y1=roi['bbox']
   assert np.array_equal(np.asarray(Image.open(full))[y0:y1,x0:x1],np.asarray(Image.open(HTML.parent/roi['path'])))
for row in list(S['views'].values())+list(C['views'].values()):
 scope=row['scope'];assert not scope.get('old_CS_W_used_for_scope',scope.get('old_CS_W_used')) and not scope.get('photo_pixels_used_as_material',scope.get('photo_RGB_used_as_material'))
 for image in row['images'].values():
  full=HTML.parent/image['path'];assert sha(full)==image['sha256']
  for cropkey in ['roi','focus']:
   roi=image[cropkey];x0,y0,x1,y1=roi['bbox']
   assert np.array_equal(np.asarray(Image.open(full))[y0:y1,x0:x1],np.asarray(Image.open(HTML.parent/roi['path'])))

screenshots=[];errors=[];bad=[]
with sync_playwright() as pw:
 browser=pw.chromium.launch(executable_path=r'C:\Program Files\Google\Chrome\Application\chrome.exe',headless=True)
 page=browser.new_page(viewport={'width':1680,'height':1160})
 page.on('pageerror',lambda e:errors.append(str(e)))
 page.on('response',lambda r:bad.append((r.status,r.url)) if r.status>=400 and 'road_vehicle_joint_dggt_r8' in r.url else None)
 page.goto('http://127.0.0.1:18765/index.html#road-vehicle-joint-dggt',wait_until='domcontentloaded')
 page.wait_for_selector('#raw-teacher-world-fusion')
 wait=lambda selector:page.wait_for_function('(s)=>{const xs=Array.from(document.querySelectorAll(s));return xs.length>0&&xs.every(x=>x.complete&&x.naturalWidth)}',arg=selector)
 wait('#r8-baseline-images img');wait('#r8-input-images img')
 assert page.locator('#r8-baseline-images figure').count()==3
 assert page.locator('#r8-baseline-frame option').count()==5
 assert '25 前向／0反向' in page.locator('#r8-baseline-proof').text_content()
 assert 'R8累计32F7B，0更新' in page.locator('#r8-owner-ledger').inner_text()
 assert page.locator('#r8-colour-guard tbody tr').count()==5
 for raw in [110,86,134,179,198]:
  page.locator('#r8-baseline-frame').select_option(str(raw))
  for channel in ['rgb','alpha']:
   page.locator('#r8-baseline-channel').select_option(channel);wait('#r8-baseline-images img')
   assert page.locator('#r8-baseline-images img').evaluate_all('xs=>xs.every(x=>x.getBoundingClientRect().width===x.naturalWidth)')
  page.locator('#r8-input-frame').select_option(str(raw));wait('#r8-input-images img')
  assert page.locator('#r8-input-images img').evaluate_all('xs=>xs.map(x=>[x.naturalWidth,x.naturalHeight])')==[[1152,768],[518,350]]
 page.locator('#r8-baseline-frame').select_option('110');page.locator('#r8-baseline-channel').select_option('rgb')
 page.locator('#r8-baseline-region').select_option('full');page.locator('#r8-baseline-zoom').select_option('2');wait('#r8-baseline-images img')
 assert page.locator('#r8-baseline-images img').evaluate_all('xs=>xs.every(x=>x.naturalWidth===1152&&x.getBoundingClientRect().width===2304)')
 page.locator('#r8-baseline-region').select_option('roi');page.locator('#r8-baseline-zoom').select_option('1');wait('#r8-baseline-images img')
 page.locator('#road-vehicle-joint-dggt').evaluate('(x)=>x.scrollIntoView({block:"start"})')
 screenshot=QA/'r8_aligned_baseline_110.png';page.screenshot(path=str(screenshot));screenshots.append(str(screenshot))
 page.locator('#r8-coverage-audit').evaluate('(x)=>{x.open=true;x.scrollIntoView({block:"start"})}');wait('#r8-coverage-audit img')
 assert page.locator('#r8-coverage-audit tbody tr').count()==5
 screenshot=QA/'r8_coverage_occlusion_audit.png';page.screenshot(path=str(screenshot));screenshots.append(str(screenshot))
 page.locator('#r8-coverage-audit').evaluate('(x)=>x.open=false')
 assert 'R7失败候选已拒绝' in page.locator('#r7-rejection-policy').inner_text()
 page.locator('#r7-photo-frame').select_option('86');assert 'R7失败候选已拒绝' in page.locator('#r7-photo-verdict').inner_text()
 page.locator('#r7-photo-frame').select_option('110')
 page.locator('#r7-black-rim-cause').evaluate('(x)=>{x.open=true;x.scrollIntoView({block:"start"})}');wait('#r7-black-rim-cause img')
 screenshot=QA/'r7_fixed_rim_cause.png';page.screenshot(path=str(screenshot));screenshots.append(str(screenshot))
 page.evaluate('location.hash="raw-teacher-world-fusion"')
 assert page.locator('#raw-teacher-world-fusion').is_visible()
 r9=page.locator('#raw-teacher-world-fusion').inner_text();assert '4173' in r9 and '1792' in r9 and '联合高斯拟合尚未运行' in r9
 assert '共享表面输入通过 · 联合拟合未运行' in page.locator('#raw-teacher-world-fusion h2').inner_text()
 assert page.locator('#r9-source-stages figure').count()==4 and page.locator('#r9-source-stages img').count()==3
 for raw in [110,86,134,179,198]:
  page.locator('#r9-source-frame').select_option(str(raw))
  for layer in ['material_hatched','material_only','source_kind','teacher_influence']:
   page.locator('#r9-current-layer').select_option(layer);wait('#r9-source-stages img')
   assert page.locator('#r9-source-stages img').evaluate_all('xs=>xs.every(x=>x.getBoundingClientRect().width===x.naturalWidth)')
  assert '不是逐点RGB混合α' in page.locator('#r9-current-domain').inner_text()
  assert page.locator('#r9-source-stages img').count()==(3 if raw in [110,86] else 1)
 page.locator('#r9-source-frame').select_option('110');page.locator('#r9-source-region').select_option('full');page.locator('#r9-source-zoom').select_option('2');wait('#r9-source-stages img')
 assert page.locator('#r9-source-stages img').evaluate_all('xs=>xs.every(x=>x.naturalWidth===1152&&x.getBoundingClientRect().width===2304)')
 page.locator('#r9-source-region').select_option('focus');page.locator('#r9-source-zoom').select_option('1');page.locator('#r9-current-layer').select_option('material_hatched');wait('#r9-source-stages img')
 screenshot=QA/'r9_continuous_input_accepted_110.png';page.screenshot(path=str(screenshot));screenshots.append(str(screenshot))
 assert '40,850' in page.locator('#r9-tail-note').text_content()
 assert not page.locator('#r9-shared-v1').evaluate('(x)=>x.open')
 page.locator('#r9-shared-v1').evaluate('(x)=>{x.open=true;x.scrollIntoView({block:"start"})}')
 assert '目视未通过' in page.locator('#r9-shared-v1-verdict').inner_text()
 for raw in [110,86,134,179,198]:
  page.locator('#r9-shared-frame').select_option(str(raw))
  for layer in ['material_hatched','material_only','source_kind','teacher_alpha']:
   page.locator('#r9-shared-layer').select_option(layer);wait('#r9-shared-images img')
  assert page.locator('#r9-shared-images img').count()==(3 if raw in [110,86] else 1)
 page.locator('#r9-shared-frame').select_option('110');page.locator('#r9-shared-layer').select_option('material_hatched');wait('#r9-shared-images img')
 screenshot=QA/'r9_shared_v1_visual_fail_110.png';page.screenshot(path=str(screenshot));screenshots.append(str(screenshot))
 page.locator('#r9-shared-zoom').select_option('2');wait('#r9-shared-images img')
 assert page.locator('#r9-shared-images img').evaluate_all('xs=>xs.every(x=>x.getBoundingClientRect().width===x.naturalWidth*2)')
 page.locator('#r9-shared-zoom').select_option('1');page.locator('#r9-shared-v1').evaluate('(x)=>x.open=false')
 page.set_viewport_size({'width':390,'height':844})
 assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),page.evaluate('[document.documentElement.scrollWidth,innerWidth]')
 assert not errors,errors
 assert not bad,bad
 browser.close()
record=dict(passed=True,checks=['R8累计32F7B、0更新；归属与五视角色彩保护测量未改RGBA','实际基线与R9共享曲面的原像素裁剪一致','RGB、Alpha、五帧输入、100%／200%与移动端通过','R7历史块与A20原HTML未变；4173 Brown与1792针孔分开','R9连续场2eaf独立数值／独立目视／主审均PASS；五视角实际图与原收据绑定','教师条件影响非RGB混合α；相似实拍主导层40850微弱尾纹素明确','初版8ad4目视FAIL保留；联合高斯未运行，真实拟合adapter就绪'],native_baseline_image_count=images,current_shared_field_sha256=C['field_sha256'],joint_fit_status='not_run',page_errors=errors,HTTP_errors=bad,screenshots=screenshots,report_frozen=False)
(QA/'qa_wiring_report.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
(QA/'qa_continuous_input_report.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(record,ensure_ascii=False))
