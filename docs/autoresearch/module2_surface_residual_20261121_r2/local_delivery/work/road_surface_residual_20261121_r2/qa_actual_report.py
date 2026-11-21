"""Inspect the delivered report without recomputing scientific results."""
from pathlib import Path
import hashlib,json
from playwright.sync_api import sync_playwright

W=Path(__file__).resolve().parent
OUT=W.parents[1]/'outputs/road_material_three_rounds_20261121'
page_path=OUT/'index.html'
errors=[]
checks=0
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True,args=['--disable-gpu','--allow-file-access-from-files'])
    page=browser.new_page(viewport=dict(width=1500,height=1000))
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto(page_path.as_uri(),wait_until='load')
    assert '7,239' in page.locator('#geometry-counts').inner_text() or '7239' in page.locator('#geometry-counts').inner_text()
    assert page.locator('#cumulative tr').count()==5
    for preset in ['final','geometry','material']:
        page.click('nav button[data-preset="'+preset+'"]')
        for raw in ['86','110','134','179','198']:
            page.select_option('#raw',raw)
            for view in ['road','joint','alpha']:
                page.select_option('#view',view)
                for mode in ['full','detail']:
                    page.click('#'+mode)
                    page.wait_for_function('[...document.querySelectorAll("#gallery img")].every(i=>i.complete&&i.naturalWidth>0)')
                    assert page.locator('#gallery img').count()==(3 if raw in ['86','110'] else 2)
                    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
                    checks+=1
    page.select_option('#raw','110');page.select_option('#view','road');page.click('#detail')
    page.click('nav button[data-preset="material"]')
    page.wait_for_function('[...document.querySelectorAll("#gallery img")].every(i=>i.complete&&i.naturalWidth>0)')
    page.screenshot(path=str(W/'qa_actual_detail.png'),full_page=True)
    page.set_viewport_size(dict(width=390,height=1000))
    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth+1')
    page.screenshot(path=str(W/'qa_actual_mobile.png'),full_page=True)
    links=page.eval_on_selector_all('a[href]','es=>es.map(e=>e.getAttribute("href"))')
    missing=[]
    for link in links:
        if not link or link.startswith(('http:','https:','#')):continue
        if not (OUT/link.split('#')[0]).is_file():missing.append(link)
    browser.close()
assert not errors,errors
assert not missing,missing
receipt=dict(passed=True,scope='actual delivered report UI and local links',view_combinations=checks,JavaScript_errors=errors,missing_links=missing,new_GPU_calls=0,index_sha256=hashlib.sha256(page_path.read_bytes()).hexdigest())
(W/'qa_actual_report.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n','utf-8')
print(json.dumps(receipt,ensure_ascii=False))
