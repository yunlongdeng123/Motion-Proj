"""UI/file-pipeline smoke only. Teacher pixels are fixtures, never student results."""
from pathlib import Path
import importlib.util,json,shutil,tarfile
from PIL import Image
from playwright.sync_api import sync_playwright

W=Path(__file__).resolve().parent
Q=W/'qa_file_pipeline';Q.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('finalizer',W/'finalizer.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
f.OUT=Q/'report';f.OUT.mkdir(exist_ok=True);f.DEST=f.OUT/'common1152';f.DEST.mkdir(exist_ok=True)
(f.OUT/'index.html').write_text('<p>DISPLAY SMOKE ONLY; NO STUDENT RESULTS.</p>','utf-8')
package=Q/'fixture_bundle';(package/'delivery').mkdir(parents=True,exist_ok=True);(package/'bench').mkdir(exist_ok=True)
source=W/'teacher/raw110/teacher_rgb.png';shutil.copy2(source,package/'delivery/fixture1152.png')
with Image.open(source)as im:im.crop((400,450,720,690)).save(package/'delivery/fixture_crop.png')
fr=dict(road='delivery/fixture1152.png',joint='delivery/fixture1152.png',C_road_crop='delivery/fixture_crop.png',S_road_crop='delivery/fixture_crop.png',C_joint_crop='delivery/fixture_crop.png',S_joint_crop='delivery/fixture_crop.png')
m=dict(teacher=dict(resolution=[1152,768],frames={'110':dict(C='delivery/fixture1152.png',S='delivery/fixture1152.png',A='delivery/fixture1152.png',C_crop='delivery/fixture_crop.png',S_crop='delivery/fixture_crop.png',A_crop='delivery/fixture_crop.png')}),models=[dict(id='UI_ONLY_R'+str(g),round='R'+str(g),name='显示测试；不是学生结果',budget={'steps':0},frames={'110':fr})for g in [1,2,3]])
(package/'delivery/latest_manifest.json').write_text(json.dumps(m,ensure_ascii=False),'utf-8');(package/'bench/metrics.json').write_text(json.dumps(dict(rows=[],blackgap_rows=[])),'utf-8')
archive=Q/'display_fixture.tar.gz'
with tarfile.open(archive,'w:gz')as t:
    for p in sorted(package.rglob('*')):
        if p.is_file():t.add(p,arcname=p.relative_to(package).as_posix())
result=f.install(archive);errors=[]
with sync_playwright()as p:
    b=p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True,args=['--disable-gpu','--allow-file-access-from-files']);page=b.new_page(viewport=dict(width=1500,height=1000));page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto((f.OUT/'index.html').as_uri(),wait_until='load')
    for g in ['0','1','2','3']:
        page.click('nav button[data-group="'+g+'"]')
        for mode in ['full','detail']:
            page.click('#'+mode);page.wait_for_function('[...document.images].every(i=>i.complete&&i.naturalWidth>0)');assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
    assert page.locator('#gallery').inner_text().count('显示测试')==2
    page.screenshot(path=str(Q/'qa_desktop.png'),full_page=True);page.set_viewport_size(dict(width=390,height=1000));assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1');page.screenshot(path=str(Q/'qa_mobile.png'),full_page=True);b.close()
assert not errors
receipt=dict(passed=True,scope='file extraction + UI wiring only; source teacher used as fixture, not a trained student',production_report_untouched=True,new_GPU_calls=0,new_training_updates=0,JavaScript_errors=errors,**result)
(W/'qa_finalizer.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n','utf-8');print(json.dumps(receipt,ensure_ascii=False))
