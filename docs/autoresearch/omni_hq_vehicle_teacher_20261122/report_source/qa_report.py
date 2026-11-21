from pathlib import Path
import json
from playwright.sync_api import sync_playwright

R=Path(__file__).resolve().parent
O=R.parents[1]/'outputs/omni_hq_vehicle_teacher_20261122'
Q=R/'report_qa';Q.mkdir(exist_ok=True)
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe', headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1000},device_scale_factor=1)
    errors=[]
    page.on('pageerror',lambda e: errors.append(str(e)))
    page.goto((O/'index.html').as_uri(),wait_until='load')
    page.wait_for_function('Array.from(document.images).every(i=>i.complete)')
    images=page.locator('img').evaluate_all('(xs)=>xs.map(x=>({src:x.getAttribute("src"),ok:x.naturalWidth>0,width:x.naturalWidth,height:x.naturalHeight}))')
    links=[]
    for href in page.locator('a[href]').evaluate_all('(xs)=>xs.map(x=>x.getAttribute("href"))'):
        if not href.startswith(('http:','https:','#')):
            links.append({'href':href,'exists':(O/href).resolve().exists()})
    desktop_overflow=page.evaluate('document.documentElement.scrollWidth>innerWidth')
    page.screenshot(path=str(Q/'desktop_top.png'))
    page.get_by_role('heading', name='固定候选：正交六视灰模',exact=True).scroll_into_view_if_needed()
    page.screenshot(path=str(Q/'desktop_grey.png'))
    page.set_viewport_size({'width':390,'height':844})
    page.evaluate('scrollTo(0,0)')
    mobile_overflow=page.evaluate('document.documentElement.scrollWidth>innerWidth')
    page.screenshot(path=str(Q/'mobile_top.png'))
    receipt={'pageerrors':errors,'images':images,'links':links,'desktop_overflow':desktop_overflow,'mobile_overflow':mobile_overflow}
    receipt['passed']=not errors and all(x['ok'] for x in images) and all(x['exists'] for x in links) and not desktop_overflow and not mobile_overflow
    (Q/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['images','links']}))
    browser.close()
