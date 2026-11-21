from pathlib import Path
import json
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parents[1]/'outputs/a20_sd15_sameview_20261122'
Q=ROOT/'report_qa';Q.mkdir(exist_ok=True)
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto((OUT/'index.html').as_uri(),wait_until='load')
    page.wait_for_function('Array.from(document.images).every(i=>i.complete)')
    images=page.locator('img').evaluate_all('(xs)=>xs.map(i=>({src:i.getAttribute("src"),ok:i.naturalWidth>0}))')
    links=[{'href':v,'exists':(OUT/v).resolve().exists()} for v in page.locator('a[href]').evaluate_all('(xs)=>xs.map(i=>i.getAttribute("href"))') if not v.startswith(('http:','https:','#'))]
    states=[]
    if page.locator('#view').count():
        for value in page.locator('#view option').evaluate_all('(xs)=>xs.map(i=>i.value)'):
            page.locator('#view').select_option(value)
            page.wait_for_function('Array.from(document.images).every(i=>i.complete)')
            states.append({'view':value,'ok':page.locator('img').evaluate_all('(xs)=>xs.every(i=>i.naturalWidth>0)')})
        page.locator('#view').select_option(index=0)
        for method in page.locator('#left-method option').evaluate_all('(xs)=>xs.map(i=>i.value)'):
            page.locator('#left-method').select_option(method)
            for view in ('novel_00_rgb','novel_03_rgb'):
                page.locator('#view').select_option(view)
                page.wait_for_function('Array.from(document.images).every(i=>i.complete)')
                states.append({'view':view,'left_method':method,'ok':page.locator('img').evaluate_all('(xs)=>xs.every(i=>i.naturalWidth>0)')})
        page.locator('#left-method').select_option(index=1)
        page.locator('#view').select_option(index=0)
    desktop_overflow=page.evaluate('document.documentElement.scrollWidth>innerWidth')
    page.evaluate('scrollTo(0,0)');page.screenshot(path=str(Q/'desktop.png'))
    page.get_by_role('heading',name='同相机原生细节对照',exact=True).scroll_into_view_if_needed()
    page.screenshot(path=str(Q/'desktop_comparison.png'))
    page.set_viewport_size({'width':390,'height':844})
    mobile_overflow=page.evaluate('document.documentElement.scrollWidth>innerWidth')
    page.evaluate('scrollTo(0,0)');page.screenshot(path=str(Q/'mobile.png'))
    receipt={'errors':errors,'images':images,'links':links,'viewer_states':states,'desktop_overflow':desktop_overflow,'mobile_overflow':mobile_overflow}
    receipt['passed']=not errors and all(x['ok'] for x in images+states) and all(x['exists'] for x in links) and not desktop_overflow and not mobile_overflow
    (Q/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':receipt['passed'],'images':len(images),'states':len(states),'errors':errors}))
    browser.close()
