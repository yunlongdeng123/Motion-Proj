"""检查真实交付图像与离线页面交互，避免实验阶段或参考图错绑。"""
from pathlib import Path
import json, time
from PIL import Image
from playwright.sync_api import sync_playwright

W=Path(__file__).resolve().parent
OUT=W.parents[1]/'outputs/vehicle_road_joint_20261121'
def main():
    started=time.monotonic();d=json.loads((OUT/'report_data.json').read_text('utf-8'))
    all_paths=[]
    for s in d['stages']:
        for raw in d['frames']:
            f=s['images'][str(raw)]
            for view in ['joint_keep','joint_delete','vehicle_only','road_only']:
                path=OUT/f[view];assert path.is_file(),path
                with Image.open(path) as im:assert im.size==(1152,768),(path,im.size)
                all_paths.append(str(path))
    for r in d['references'].values():
        for value in r.values():
            if isinstance(value,str):assert (OUT/value).is_file(),value
    errors=[];cases=0
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True)
        page=browser.new_page(viewport={'width':1680,'height':1100},device_scale_factor=1)
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto((OUT/'index.html').as_uri());page.wait_for_function('typeof D !== "undefined"')
        page.select_option('#left',d['stages'][0]['id'])
        for s in d['stages']:
            page.select_option('#right',s['id'])
            for raw in d['frames']:
                page.select_option('#frame',str(raw))
                for view in ['joint_keep','joint_delete','vehicle_only','road_only']:
                    page.select_option('#view',view)
                    page.wait_for_function("['leftVisual','rightVisual'].every(id=>{const i=document.querySelector('#'+id+' img');return i&&i.complete&&i.naturalWidth===1152;})")
                    observed=page.locator('#rightVisual img').get_attribute('src')
                    assert observed==s['images'][str(raw)][view],(s['id'],raw,view,observed)
                    cases+=1
        page.select_option('#frame',str(d.get('default_frame',110)))
        page.select_option('#view','joint_delete')
        page.locator('#rightVisual img').wait_for()
        page.screenshot(path=str(W/'qa_desktop.png'),full_page=True)
        page.click('#crop');page.wait_for_timeout(300)
        page.screenshot(path=str(W/'qa_crop.png'),full_page=True)
        page.set_viewport_size({'width':390,'height':844})
        page.wait_for_timeout(200)
        overflow=page.evaluate('document.documentElement.scrollWidth>innerWidth')
        assert not overflow,'mobile document overflow'
        page.screenshot(path=str(W/'qa_mobile.png'),full_page=True)
        browser.close()
    assert not errors,errors
    result=dict(passed=True,images=len(all_paths),interaction_cases=cases,JS_errors=errors,mobile_overflow=overflow,seconds=time.monotonic()-started)
    (W/'qa_report.json').write_text(json.dumps(result,indent=2)+'\n','utf-8')
    print(json.dumps(result))
if __name__=='__main__':main()
