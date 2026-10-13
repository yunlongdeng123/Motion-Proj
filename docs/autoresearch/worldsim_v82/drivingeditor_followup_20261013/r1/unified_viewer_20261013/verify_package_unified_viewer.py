from pathlib import Path
import datetime,json,re,subprocess,zipfile,shutil
W=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-12/w');O=W/'outputs';B=W/'work/drivingeditor_resume/unified_viewer_20261013'
t=(O/'DrivingEditor_final_report.html').read_text(encoding='utf-8')
payload=re.search(r'<script id="scene-data" type="application/json">(.*?)</script>',t,re.S).group(1);d=json.loads(payload)
assert list(d)==['silver','black','red']
assert list(d['silver']['arms'])==['C00','G10','N01','GN11']
assert all(list(d[s]['arms'])==['GN11'] for s in ['black','red'])
count=0
for scene in d.values():
    for a in scene['arms'].values():
        for mode,frames in a.items():
            assert len(frames)==49 and len({f['id'] for f in frames})==49
            for f in frames:
                assert ('/observed/' in f['images']['static']) if mode=='observed' else ('/static_time/' in f['images']['static'])
                for src in f['images'].values():assert (O/src).is_file();count+=1
assert count==2646
assert "scene:'silver',mode:'observed',i:10,branch:'static'" in t
code=re.findall(r'<script>(.*?)</script>',t,re.S)[0]
js=B/'unified_viewer.js';js.write_text(code,encoding='utf-8')
r=subprocess.run(['C:/Program Files/nodejs/node.exe','--check',str(js)],capture_output=True,text=True,check=True)
old=(O/'drivingeditor_four_arm_30k/index.html').read_text(encoding='utf-8')
assert "branch:{observed:'static',time:'static'}" in old
old_js=re.findall(r'<script>(.*?)</script>',old,re.S)[0]
oj=B/'four_arm_updated.js';oj.write_text(old_js,encoding='utf-8')
subprocess.run(['C:/Program Files/nodejs/node.exe','--check',str(oj)],capture_output=True,text=True,check=True)
receipt={'status':'STATIC_ASSET_AND_JAVASCRIPT_CONTRACT_PASS','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'default':'observed_static','scenes':['silver','black','red'],'silver_arms':['C00','G10','N01','GN11'],'other_scene_arms':['GN11'],
 'image_references_checked':2646,'old_four_arm_default_static':True,'node_syntax_checks':2,'new_training_steps':0,
 'browser_interaction_verified_this_revision':False,
 'browser_limitation':'Browser tool blocks file: protocol; no workaround attempted. Local source, resources and JS syntax checked instead.',
 'old_rendered_PNG_unchanged':True,'human_GN11_verdict':None}
(O/'DrivingEditor_unified_viewer_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(B/'validation.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
qa=O/'DrivingEditor_final_browser_QA.json';prior=json.loads(qa.read_text(encoding='utf-8-sig'))
if prior.get('status')=='ROOT_ACTUAL_HUB_BROWSER_QA_PASS':receipt['historical_previous_hub_browser_QA']=prior
qa.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
dest=O/'DrivingEditor_final_reports.zip';backup=B/'DrivingEditor_final_reports.before_unified.zip'
if not backup.exists():shutil.copy2(dest,backup)
folders=['drivingeditor_four_arm_30k','drivingeditor_GMC_GN11_30k','drivingeditor_red_GN11_30k']
files=[p for folder in folders for p in (O/folder).rglob('*') if p.is_file() and p.suffix not in {'.zip','.pth','.tar','.gz'}]
files += [O/n for n in ['DrivingEditor_final_report.html','DrivingEditor_final_summary.json','DrivingEditor_resource_and_speed.png','DrivingEditor_resource_and_speed.pdf','DrivingEditor_final_browser_QA.json','DrivingEditor_shutdown_receipt.json','DrivingEditor_automation_paused.json','DrivingEditor_unified_viewer_receipt.json']]
tmp=B/'DrivingEditor_final_reports.updated.zip'
with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for p in files:z.write(p,p.relative_to(O).as_posix())
with zipfile.ZipFile(tmp) as z:
    assert z.testzip() is None
    assert z.read('DrivingEditor_final_report.html')==(O/'DrivingEditor_final_report.html').read_bytes()
    assert sum(n.endswith('.png') and '/assets/' in n for n in z.namelist())==2646
    assert len(z.namelist())==len(set(z.namelist()))
tmp.replace(dest)
pack=json.loads((O/'DrivingEditor_offline_package_receipt.json').read_text(encoding='utf-8'))
pack.update(bytes=dest.stat().st_size,files=len(files),canonical_viewer='DrivingEditor_final_report.html',default_camera_mode='observed',default_branch='static',unified_revision=True,archive_decompression_passed=True,original_report_entries_unchanged=False,original_rendered_assets_unchanged=True)
(O/'DrivingEditor_offline_package_receipt.json').write_text(json.dumps(pack,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':receipt['status'],'assets':count,'ZIP_bytes':dest.stat().st_size,'browser_interaction_verified_this_revision':False}))
