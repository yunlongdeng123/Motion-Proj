"""补查新增Alpha视图、参考缺失处理、局部画布与旧报告入口。"""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright
W=Path(__file__).resolve().parent;ROOT=W.parents[1];O=ROOT/'outputs/vehicle_road_joint_20261121'
d=json.loads((O/'report_data.json').read_text('utf-8'))
assert len(d['stages'])==7 and len(d['metrics'])==49
assert len({(r['stage'],r['raw'],r['role']) for r in d['metrics']})==49
errors=[];cases=0
with sync_playwright() as p:
    b=p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True)
    page=b.new_page(viewport={'width':1680,'height':1100});page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto((O/'index.html').as_uri())
    for s in d['stages']:
        page.select_option('#right',s['id'])
        for raw in d['frames']:
            page.select_option('#frame',str(raw))
            for view in ['vehicle_alpha_delete','road_alpha']:
                path=s['images'][str(raw)][view];assert path and (O/path).is_file()
                page.select_option('#view',view)
                page.wait_for_function("(()=>{let i=document.querySelector('#rightVisual img');return i&&i.complete&&i.naturalWidth===1152;})()")
                assert page.locator('#rightVisual img').get_attribute('src')==path
                cases+=1
    page.select_option('#frame','134');page.select_option('#view','joint_delete')
    page.select_option('#reference','road_teacher');page.wait_for_timeout(50)
    assert page.locator('#refVisual .empty').count()==1
    page.select_option('#reference','auto');page.wait_for_timeout(50)
    assert page.locator('#refTitle').inner_text()=='原始真实观测'
    page.select_option('#right','handoff_fixed_guarded')
    page.click('#crop');page.wait_for_function("document.querySelectorAll('.visual canvas').length===3")
    page.locator('section').nth(1).screenshot(path=str(W/'qa_viewer.png'))
    page.click('#native');page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(100)
    assert not page.evaluate('document.documentElement.scrollWidth>innerWidth')
    old=b.new_page();old.goto((ROOT/'outputs/road_material_three_rounds_20261121/index.html').as_uri())
    old.wait_for_selector('#vehicle-road-report-entry');assert old.locator('#left option').count()==55
    entry=old.locator('#vehicle-road-report-entry a').first.get_attribute('href')
    assert entry=='../vehicle_road_joint_20261121/index.html'
    b.close()
assert not errors,errors
r=dict(passed=True,new_alpha_cases=cases,stages=7,fixed_domain_metric_rows=49,local_crop_canvases=True,explicit_missing_teacher=True,original_report_records=55,JS_errors=errors)
(W/'qa_extras.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n','utf-8');print(json.dumps(r))
