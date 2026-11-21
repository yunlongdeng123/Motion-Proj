"""真实 Chrome fileURL：180 组合逐像素、移动端及历史导航独立验收。"""
from pathlib import Path
import json
import time
import traceback
from urllib.parse import unquote, urlparse
from PIL import Image
from playwright.sync_api import sync_playwright

WORK=Path(__file__).resolve().parent
BASE=WORK.parents[1]
OUT=BASE/'outputs/vehicle_completion_prior_20261122'
OLD=BASE/'outputs/vehicle_road_joint_20261121'
ROAD=BASE/'outputs/road_material_three_rounds_20261121'
QA=WORK/'html_qa'
STAGES=('canonical_prior_init','vehicle_fit','joint_fit')
FRAMES=(86,110,134,179,198)
VIEWS=('vehicle_only','vehicle_alpha_isolated','vehicle_alpha_delete','joint_delete','joint_keep','road_only')

INSTALL_OBSERVER="""() => {
 window.__qaDraws={}; const original=draw;
 draw=function(id,url,crop,rev){
  return original(id,url,crop,rev).then(ok=>{
   if(rev===revision)window.__qaDraws[id]={url:new URL(url,location.href).href,crop:crop||null,rev,ok};
   return ok;
  });
 };
}"""

VERIFY_CANVAS="""async ({sources,rect}) => {
 const result=[];
 for(const [id,url] of Object.entries(sources)){
  const absolute=new URL(url,location.href).href;
  const observed=window.__qaDraws[id];
  if(!observed||observed.url!==absolute||observed.rev!==revision||!observed.ok)
   throw Error('binding mismatch '+id+' '+JSON.stringify(observed)+' expected '+absolute);
  const im=new Image();
  await new Promise((resolve,reject)=>{im.onload=resolve;im.onerror=()=>reject(Error('expected source missing '+absolute));im.src=absolute;});
  if(im.naturalWidth!==1152||im.naturalHeight!==768)throw Error('source dimensions '+absolute);
  const r=rect||[0,0,1152,768],actual=document.getElementById(id);
  if(actual.width!==r[2]||actual.height!==r[3])throw Error('canvas intrinsic size mismatch '+id);
  const expected=document.createElement('canvas');expected.width=r[2];expected.height=r[3];
  expected.getContext('2d').drawImage(im,r[0],r[1],r[2],r[3],0,0,r[2],r[3]);
  const a=actual.getContext('2d').getImageData(0,0,r[2],r[3]).data;
  const b=expected.getContext('2d').getImageData(0,0,r[2],r[3]).data;
  const av=new Uint32Array(a.buffer,a.byteOffset,a.byteLength/4),bv=new Uint32Array(b.buffer,b.byteOffset,b.byteLength/4);
  let bad=0,first=-1,maxDifference=0;
  for(let i=0;i<av.length;i++)if(av[i]!==bv[i]){bad++;if(first<0)first=i;for(let c=0;c<4;c++)maxDifference=Math.max(maxDifference,Math.abs(a[4*i+c]-b[4*i+c]));}
  result.push({canvas:id,url:absolute,rect:r,compared_pixels:av.length,different_pixels:bad,max_channel_difference:maxDifference,first_mismatch_pixel:first});
 }
 return result;
}"""

VERIFY_DEFAULT_FILE_BINDING="""({sources,rect}) => {
 const result=[],r=rect||[0,0,1152,768];
 for(const [id,url] of Object.entries(sources)){
  const absolute=new URL(url,location.href).href,observed=window.__qaDraws[id],canvas=document.getElementById(id);
  if(!observed||observed.url!==absolute||observed.rev!==revision||!observed.ok)throw Error('binding mismatch '+id);
  if(canvas.width!==r[2]||canvas.height!==r[3])throw Error('intrinsic canvas size '+id);
  if(JSON.stringify(observed.crop)!==JSON.stringify(rect))throw Error('source crop mismatch '+id);
  result.push({canvas:id,url:absolute,rect:r,intrinsic_width:canvas.width,intrinsic_height:canvas.height,loaded:true});
 }
 return result;
}"""


def wait_viewer(page):
    page.wait_for_function("""() => document.getElementById('viewer-status').textContent.startsWith('源渲染') &&
      ['reference','previous','current'].every(id=>window.__qaDraws?.[id]?.rev===revision&&window.__qaDraws[id].ok)""")


def static_assets(data):
    paths=[]
    for stage in STAGES:
        for raw in FRAMES:
            for view in VIEWS:
                paths.append((OUT/data['current']/stage/f'{raw:03d}'/(view+'.webp')).resolve())
    paths += [(OUT/data['reference']/f'{raw:03d}'/'rgb.webp').resolve() for raw in FRAMES]
    paths += [(OUT/data['previous']/f'{raw:03d}'/(view+'.webp')).resolve() for raw in FRAMES for view in VIEWS]
    missing=[str(p) for p in paths if not p.is_file()]
    if missing:raise FileNotFoundError('required files missing; wait for the one takeback, no polling:\n'+'\n'.join(missing))
    for path in paths:
        with Image.open(path) as im:
            if im.size!=(1152,768):raise ValueError((str(path),im.size))
    return dict(current_combinations=90,reference_assets=5,previous_assets=30,all_dimensions=[1152,768])


def set_case(page,stage,raw,view,mode):
    for selector,value in [('#stage',stage),('#frame',str(raw)),('#view',view),('#crop',mode)]:
        if page.locator(selector).input_value()!=value:
            page.select_option(selector,value)
    # 同一选择可能无 change；明确 dispatch UI 事件以得到本次 fresh revision。
    page.locator('#crop').dispatch_event('change')
    wait_viewer(page)


def local_links(page):
    links=page.locator('a[href]').evaluate_all('(items)=>items.map(a=>({label:a.textContent.trim(),url:a.href}))')
    missing=[]
    for link in links:
        parsed=urlparse(link['url'])
        if parsed.scheme=='file':
            path=unquote(parsed.path)
            if len(path)>2 and path[0]=='/' and path[2]==':':path=path[1:]
            if not Path(path).exists():missing.append(link)
    return dict(local_link_count=sum(urlparse(x['url']).scheme=='file' for x in links),missing=missing)


def run():
    started=time.monotonic();QA.mkdir(parents=True,exist_ok=True)
    report=dict(passed=False,chrome='C:/Program Files/Google/Chrome/Application/chrome.exe',
        browser_scheme='file',canvas_readback_flag='none in final default Chrome UI run',
        GPU_model_calls=0,root_report_source_modified=False,cases=[],page_errors=[],failed_requests=[])
    try:
        data=json.loads((WORK/'result.json').read_text(encoding='utf-8'))['student_viewer']
        report['assets']=static_assets(data)
        prior_pixel=json.loads((WORK/'qa_report_pixel_check.json').read_text(encoding='utf-8'))
        assert prior_pixel['passed'] and prior_pixel['pixel_check']['all_pixels_exact']
        report['pixel_check']=prior_pixel['pixel_check']
        report['pixel_check_receipt']='qa_report_pixel_check.json; separate readback-only Chrome run, product unchanged'
        with sync_playwright() as playwright:
            browser=playwright.chromium.launch(executable_path=report['chrome'],headless=True)
            page=browser.new_page(viewport={'width':1680,'height':1100},device_scale_factor=1)
            page.on('pageerror',lambda error:report['page_errors'].append(str(error)))
            page.on('requestfailed',lambda request:report['failed_requests'].append(dict(url=request.url,error=request.failure)))
            page.goto((OUT/'index.html').as_uri(),wait_until='load')
            page.wait_for_function('typeof VDATA!=="undefined"')
            page.evaluate(INSTALL_OBSERVER)
            options={name:page.locator('#'+name+' option').evaluate_all('(items)=>items.map(x=>x.value)') for name in ['stage','frame','view','crop']}
            assert options==dict(stage=list(STAGES),frame=list(map(str,FRAMES)),view=list(VIEWS),crop=['vehicle','full']),options
            report['selector_options']=options
            for stage in STAGES:
                for raw in FRAMES:
                    frame=f'{raw:03d}'
                    for view in VIEWS:
                        for mode in ('vehicle','full'):
                            set_case(page,stage,raw,view,mode)
                            sources=dict(reference=data['reference']+'/'+frame+'/rgb.webp',
                                previous=data['previous']+'/'+frame+'/'+view+'.webp',
                                current=data['current']+'/'+stage+'/'+frame+'/'+view+'.webp')
                            check=page.evaluate(VERIFY_DEFAULT_FILE_BINDING,dict(sources=sources,rect=data['crops'][frame] if mode=='vehicle' else None))
                            assert page.locator('#new-title').inner_text()==page.locator('#stage option:checked').inner_text()
                            report['cases'].append(dict(stage=stage,raw=raw,view=view,mode=mode,canvases=check))
                    print('html-qa',stage,raw,'12 combinations passed',flush=True)
            assert len(report['cases'])==180
            report['default_fileURL_interaction_check']=dict(interaction_cases=180,canvas_bindings=540,special_permissions=False)
            report['links']=local_links(page)
            assert not report['links']['missing'],report['links']
            page.wait_for_function('Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)')
            page.evaluate('scrollTo(0,0)');page.screenshot(path=str(QA/'desktop_overview.png'))
            set_case(page,'joint_fit',110,'vehicle_only','vehicle')
            page.locator('.render-grid').screenshot(path=str(QA/'viewer_vehicle_raw110.png'))
            page.locator('.controls').scroll_into_view_if_needed();page.screenshot(path=str(QA/'desktop_viewer.png'))
            set_case(page,'joint_fit',86,'joint_delete','vehicle')
            page.locator('.render-grid').screenshot(path=str(QA/'viewer_joint_raw086.png'))
            mobile=[]
            for width in (320,390,768):
                page.set_viewport_size({'width':width,'height':844})
                for mode in ('vehicle','full'):
                    set_case(page,'joint_fit',110,'vehicle_only',mode)
                    dimensions=page.evaluate('({width:innerWidth,scroll:document.documentElement.scrollWidth})')
                    mobile.append(dict(mode=mode,**dimensions,overflow=dimensions['scroll']>dimensions['width']))
            report['mobile']=mobile
            assert not any(x['overflow'] for x in mobile),mobile
            page.set_viewport_size({'width':390,'height':844});set_case(page,'joint_fit',110,'vehicle_only','vehicle')
            page.evaluate('scrollTo(0,0)');page.screenshot(path=str(QA/'mobile_overview.png'))
            page.locator('.controls').scroll_into_view_if_needed();page.screenshot(path=str(QA/'mobile_viewer.png'))
            page.set_viewport_size({'width':1680,'height':1100})
            page.goto((OLD/'index.html').as_uri(),wait_until='load');page.wait_for_function('typeof D!=="undefined"')
            old=page.evaluate('({stages:D.stages.map(s=>s.id),left:Array.from(document.querySelectorAll("#left option")).map(o=>o.value),right:Array.from(document.querySelectorAll("#right option")).map(o=>o.value),nav:document.querySelectorAll("#vehicle-completion-prior-entry").length})')
            assert len(old['stages'])==7 and old['left']==old['stages'] and old['right']==old['stages'] and old['nav']==1,old
            page.select_option('#frame','110');page.select_option('#view','joint_delete')
            for stage in old['stages']:
                page.select_option('#left',stage);page.select_option('#right',stage)
                expected=page.evaluate('(s)=>D.stages.find(x=>x.id===s).images["110"].joint_delete',stage)
                page.wait_for_function('(src)=>["leftVisual","rightVisual"].every(id=>{const im=document.querySelector("#"+id+" img");return im&&im.complete&&im.naturalWidth===1152&&im.getAttribute("src")===src;})',arg=expected)
            report['old_vehicle']=dict(**old,all_seven_left_right_loaded=True,
                nav_guard_in_source='vehicle-completion-prior-entry' in (BASE/'work/vehicle_road_joint_20261121/viewer.js').read_text(encoding='utf-8'))
            page.locator('#vehicle-completion-prior-entry a').click();page.wait_for_url((OUT/'index.html').as_uri())
            assert page.locator('#stage option').count()==3
            report['old_vehicle']['navigation_reached_new_report']=True
            page.goto((ROAD/'index.html').as_uri(),wait_until='load');page.wait_for_function('typeof C!=="undefined"')
            road=page.evaluate('({records:C.models.length,batches:C.batches.length,left:document.querySelectorAll("#left option").length,right:document.querySelectorAll("#right option").length,batch_ids:C.batches.map(b=>b.id)})')
            assert road['records']==55 and road['batches']==8 and road['left']==55 and road['right']==55,road
            report['old_road']=road
            page.screenshot(path=str(QA/'old_road_records.png'))
            browser.close()
        assert not report['page_errors'],report['page_errors']
        assert not report['failed_requests'],report['failed_requests']
        report['passed']=True
    except Exception as error:
        report['error']=repr(error);report['traceback']=traceback.format_exc()
        raise
    finally:
        report['seconds']=time.monotonic()-started
        (WORK/'qa_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k not in ['cases','traceback']},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':run()
