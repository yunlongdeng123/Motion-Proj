"""Isolated report/file validation, not a scientific experiment."""
from pathlib import Path
import importlib.util,json,shutil,tarfile
from playwright.sync_api import sync_playwright

W=Path(__file__).resolve().parent;ROOT=W.parents[1];Q=W/'qa_file_pipeline';Q.mkdir(exist_ok=True)
s=importlib.util.spec_from_file_location('surface',W/'finalizer.py');f=importlib.util.module_from_spec(s);s.loader.exec_module(f);f=f.D
f.F.OUT=Q/'report';f.F.OUT.mkdir(exist_ok=True);f.F.DEST=f.F.OUT/'surface_adaptive';f.F.DEST.mkdir(exist_ok=True)
(f.F.OUT/'index.html').write_text('DISPLAY SMOKE; NO EXPERIMENTAL RESULTS','utf-8')
B=Q/'bundle';(B/'delivery').mkdir(parents=True,exist_ok=True);(B/'bench').mkdir(exist_ok=True)
fixture=ROOT/'work/road_common1152_20261121/qa_file_pipeline/fixture_bundle/delivery'
for n in ['fixture1152.png','fixture_crop.png']:shutil.copy2(fixture/n,B/'delivery'/n)
frames={str(raw):dict(road='delivery/fixture1152.png',joint='delivery/fixture1152.png',alpha='delivery/fixture1152.png',C_road_crop='delivery/fixture_crop.png',S_road_crop='delivery/fixture_crop.png',C_joint_crop='delivery/fixture_crop.png',S_joint_crop='delivery/fixture_crop.png')for raw in [86,110,134,179,198]}
stages=['parent','activated','dense_init','coverage','appearance']
models=[dict(id=k,stage=k,round='4',name='显示链路检查（非实验）'+k,budget='test fixture',frames=frames,point_counts=dict(total=595029,active=161264,activated_existing=0,appended_new_geometry=0))for k in stages]
m=dict(teacher=dict(resolution=[1152,768],frames={'110':dict(A='delivery/fixture1152.png',C='delivery/fixture1152.png',S='delivery/fixture1152.png',C_crop='delivery/fixture_crop.png',S_crop='delivery/fixture_crop.png')}),models=models)
def bundle(name,manifest):
    (B/'delivery/latest_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False),'utf-8')
    (B/'bench/metrics.json').write_text(json.dumps(dict(rows=[],blackgap_rows=[])),'utf-8')
    target=Q/name
    with tarfile.open(target,'w:gz')as t:
        for p in sorted(B.rglob('*')):
            if p.is_file():t.add(p,arcname=p.relative_to(B).as_posix())
    return target
result=f.install(bundle('valid.tar.gz',m));before=(f.F.OUT/'index.html').read_bytes()
bad=json.loads(json.dumps(m));bad['models'][-1]['point_counts']['appended_new_geometry']=892
try:f.install(bundle('mislabel.tar.gz',bad));raise AssertionError('activation was accepted as appended geometry')
except ValueError as e:assert 'append count'in str(e)
assert (f.F.OUT/'index.html').read_bytes()==before
errors=[]
with sync_playwright()as p:
    b=p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True,args=['--disable-gpu','--allow-file-access-from-files']);page=b.new_page(viewport=dict(width=1500,height=1000));page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto((f.F.OUT/'index.html').as_uri(),wait_until='load')
    for preset in ['final','geometry','material']:
        page.click('nav button[data-preset="'+preset+'"]')
        for view in ['road','joint','alpha']:
            page.select_option('#view',view)
            for mode in ['full','detail']:
                page.click('#'+mode);page.wait_for_function('[...document.images].every(i=>i.complete&&i.naturalWidth>0)');assert page.locator('#gallery img').count()==3;assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
    page.screenshot(path=str(Q/'qa_desktop.png'),full_page=True);page.set_viewport_size(dict(width=390,height=1000));assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1');page.screenshot(path=str(Q/'qa_mobile.png'),full_page=True);b.close()
assert not errors
receipt=dict(passed=True,scope='isolated file/UI fixture only; not new point or rendering experiment',mislabel_892_rejected=True,atomic_page_preserved_on_bad_counts=True,JavaScript_errors=errors,production_report_untouched=True,new_GPU_calls=0,**result)
(W/'qa_delivery.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n','utf-8');print(json.dumps(receipt,ensure_ascii=False))
