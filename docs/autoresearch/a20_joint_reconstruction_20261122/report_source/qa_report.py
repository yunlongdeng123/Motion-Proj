"""真实浏览器验证本地报告所有实际阶段/相机/通道，截图desktop/mobile。"""
from pathlib import Path
from urllib.parse import unquote
import json
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parents[1]/'outputs/a20_joint_reconstruction_20261122'
Q=ROOT/'report_qa';Q.mkdir(exist_ok=True)
data=json.loads((ROOT/'result.json').read_text(encoding='utf-8'))
def wait(page):page.wait_for_function('Array.from(document.images).every(i=>i.complete)')
def images_ok(page):return page.locator('img').evaluate_all('(xs)=>xs.every(i=>i.naturalWidth>0)')
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto((OUT/'index.html').as_uri(),wait_until='load');wait(page)
    states=[]
    selectable=page.locator('#group option').evaluate_all('(xs)=>xs.map(x=>x.value)')
    for group in data['groups']:
        if group['id'] not in selectable:
            states.append(dict(group=group['id'],not_started_not_in_selector=True,ok=group['state'] in ('pending','not_started')));continue
        page.locator('#group').select_option(group['id'])
        if group['state']=='pending':
            states.append(dict(group=group['id'],pending=True,ok=page.locator('.image-body img').count()==0));continue
        for camera in data['cameras']:
            page.locator('#camera').select_option(camera['id'])
            if camera['id'].startswith('raw'):
                page.locator('#left-mode').select_option('shared');page.locator('#right-mode').select_option('isolated')
            for channel in ['rgb','alpha','depth','normal','grey']:
                page.locator('#channel').select_option(channel);wait(page)
                states.append(dict(group=group['id'],camera=camera['id'],channel=channel,ok=images_ok(page)))
    required_geometry=[]
    for group in data['groups']:
        if group['id'] not in ('initial','r2_G40','r3_G40','r3_ablation'):continue
        for i in range(6):
            camera='novel_'+str(i).zfill(2)
            for channel in ('depth','normal'):
                required_geometry.append(dict(group=group['id'],camera=camera,channel=channel,ok=bool(group['views'].get(camera,{}).get('isolated',{}).get(channel))))
                if group['id'] in ('r3_G40','r3_ablation'):required_geometry.append(dict(group=group['id'],camera=camera,channel='pure_T_'+channel,ok=bool(group['views'].get(camera,{}).get('teacher',{}).get(channel))))
    page.locator('#group').select_option(data['current_group']);page.locator('#native-view').click();wait(page)
    links=[]
    for href in page.locator('a[href]').evaluate_all('(xs)=>xs.map(i=>i.getAttribute("href"))'):
        if href.startswith(('https:','http:','#')):continue
        links.append(dict(href=href,exists=(OUT/unquote(href)).resolve().is_file()))
    desktop_overflow=page.evaluate('document.documentElement.scrollWidth>innerWidth')
    page.evaluate('scrollTo(0,0)');page.screenshot(path=str(Q/'desktop.png'))
    page.locator('#left-image').scroll_into_view_if_needed();page.screenshot(path=str(Q/'shared_isolated.png'))
    page.locator('#geometry-view').click();wait(page);page.locator('#left-image').scroll_into_view_if_needed();page.screenshot(path=str(Q/'depth_normal.png'))
    page.set_viewport_size({'width':390,'height':844});page.locator('#native-view').click();wait(page)
    mobile_overflow=page.evaluate('document.documentElement.scrollWidth>innerWidth')
    page.evaluate('scrollTo(0,0)');page.screenshot(path=str(Q/'mobile.png'))
    old=browser.new_page();old.goto((OUT.parent/'a20_sd15_sameview_20261122/index.html').as_uri(),wait_until='load')
    old_option_count=old.locator('#left-method option').count()
    forward_count=old.locator('a[href="../a20_joint_reconstruction_20261122/index.html"]').count()
    receipt=dict(errors=errors,states=states,required_expected_depth_and_normal=required_geometry,links=links,
        desktop_overflow=desktop_overflow,mobile_overflow=mobile_overflow,
        historical_methods=old_option_count,historical_forward_links=forward_count,
        model_frames_modified=False,AutoDL_GPU_or_runtime_operations=False)
    receipt['passed']=not errors and all(x['ok'] for x in states+required_geometry) and all(x['exists'] for x in links) and not desktop_overflow and not mobile_overflow and old_option_count==3 and forward_count==1
    (Q/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=receipt['passed'],states=len(states),geometry_channels=len(required_geometry),errors=errors,desktop_overflow=desktop_overflow,mobile_overflow=mobile_overflow,historical_methods=old_option_count)))
    browser.close()
