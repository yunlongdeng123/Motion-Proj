"""本地报告的显示、原生裁片与源数值绑定验收；不计算科研指标。"""
from pathlib import Path
import argparse,hashlib,itertools,json,math
from playwright.sync_api import sync_playwright

def main(root):
    root=Path(root).resolve();data=json.loads((root/'evaluation/results.json').read_text('utf-8'))
    method_groups=dict(QUERY=['F0','G_RGB','G_DETAIL','G_NUMERIC'],BUILD=['F0','G_BUILD_FIELD','G_BUILD_NUMERIC'])
    digest=hashlib.sha256((root/'index.html').read_bytes()).hexdigest();checks=[];errors=[]
    def check(name,ok,detail=None):checks.append(dict(check=name,passed=bool(ok),detail=detail))
    def number(cell,value,digits):
        if value is None:return cell=='N/A'
        try:return abs(float(cell)-value)<=.50001*10**(-digits)
        except ValueError:return False
    with sync_playwright()as p:
        browser=p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True,args=['--disable-gpu','--allow-file-access-from-files'])
        page=browser.new_page(viewport=dict(width=1600,height=1100),device_scale_factor=1)
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto((root/'index.html').as_uri(),wait_until='load')
        runs=[]
        for phase,methods in method_groups.items():
            page.select_option('#phase',phase)
            controls=page.evaluate("Object.fromEntries(['raw','view','region','split','sigma'].map(k=>[k,[...document.getElementById(k).options].map(x=>x.value)]))")
            check('fixed_control_domains_'+phase,controls==dict(raw=['110','86','134','179','198'],view=['road','joint'],region=['C','S'],split=['seen','held']if phase=='QUERY'else['BUILD_fit_all_qualified'],sigma=['1','2','4']))
            for raw,view,source,split,sigma in itertools.product(controls['raw'],controls['view'],controls['region'],controls['split'],controls['sigma']):
                for key,value in [('raw',raw),('view',view),('region',source),('split',split),('sigma',sigma)]:page.select_option('#'+key,value)
                page.wait_for_function("[...document.querySelectorAll('img')].every(x=>x.complete&&x.naturalWidth>0)")
                dom=page.evaluate("({cells:[...document.querySelectorAll('#metrics tr')].slice(1).map(r=>[...r.cells].map(c=>c.textContent)),images:[...document.querySelectorAll('#crops img')].map(x=>({w:x.naturalWidth,h:x.naturalHeight,cw:x.getBoundingClientRect().width,ch:x.getBoundingClientRect().height})),width:document.documentElement.scrollWidth,view:innerWidth,domain:document.getElementById('domain').textContent})")
                rows=[r for r in data['rows']if r['raw']==int(raw)and r['source']==source and r['split']==split and r['scale']==int(sigma)and r['arm']in methods and r.get('phase','QUERY')==phase]
                tag='/'.join([phase,raw,view,source,split,sigma]);check('row_count_'+tag,len(rows)==len(dom['cells']))
                cells_ok=True
                for row,c in zip(rows,dom['cells']):
                    cells_ok&=c[1]==str(row['core_n'])
                    for cell,key,n in zip(c[2:7],['RGB_MAE','HF_cosine','HF_energy_ratio','HF_relative_RMSE','flat_pred_RMS'],[5,4,2,3,5]):cells_ok&=number(cell,row.get(key),n)
                    cells_ok&=number(c[8],row['Sobel_RMSE'],5)
                check('numeric_source_binding_'+tag,cells_ok)
                check('native_crop_unscaled_'+tag,all(x['w']==x['cw']and x['h']==x['ch']for x in dom['images']))
                check('no_body_overflow_'+tag,dom['width']<=dom['view']+1)
                runs.append(tag)
        page.select_option('#phase','BUILD')
        for key,value in [('raw','110'),('view','road'),('region','C'),('split','BUILD_fit_all_qualified'),('sigma','1')]:page.select_option('#'+key,value)
        page.wait_for_function("[...document.images].every(x=>x.complete&&x.naturalWidth>0)")
        page.screenshot(path=str(root/'qa_desktop.png'),full_page=True)
        for width in [1280,768,390,320]:
            page.set_viewport_size(dict(width=width,height=1000))
            check('responsive_'+str(width),page.evaluate('document.documentElement.scrollWidth<=innerWidth+1'))
        page.set_viewport_size(dict(width=390,height=1000));page.screenshot(path=str(root/'qa_mobile.png'),full_page=True)
        check('dark_scheme',page.evaluate("getComputedStyle(document.documentElement).colorScheme.includes('dark')"))
        links=page.eval_on_selector_all('a[href]','xs=>xs.map(x=>x.getAttribute("href"))')
        check('report_local_links',all((root/link).is_file()for link in links if not link.startswith(('http:','https:','#'))))
        browser.close()
    check('no_javascript_errors',not errors,errors)
    check('index_unchanged',hashlib.sha256((root/'index.html').read_bytes()).hexdigest()==digest)
    result=dict(passed=all(x['passed']for x in checks),combinations=len(runs),checks=checks,errors=errors,
        index_SHA256=digest,scope='display and source binding only; no image quality verdict, GPU, native calls or optimization',visual_review_pending=True)
    (root/'qa_browser.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'],combinations=len(runs),checks=len(checks),failures=[x for x in checks if not x['passed']]),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');main(p.parse_args().root)
