"""Render the final local report and exercise its real RGB-only frame controls."""
import html
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT.parents[1]/'work/v44_full5_report_qa'
CHROME = Path('C:/Program Files/Google/Chrome/Application/chrome.exe')


def run(url, stem, budget):
    result = subprocess.run([str(CHROME),'--headless=new','--no-sandbox','--disable-gpu','--allow-file-access-from-files',
        '--hide-scrollbars','--no-first-run','--window-size=1440,1100','--virtual-time-budget='+str(budget),
        '--user-data-dir='+str(WORK/'chrome_profile'), '--screenshot='+str(WORK/(stem+'.png')), '--dump-dom',url],
        capture_output=True,timeout=45,creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode==0
    dom=result.stdout.decode('utf-8',errors='replace')
    (WORK/(stem+'.html')).write_text(dom,encoding='utf-8')
    return dom


def main():
    WORK.mkdir(parents=True,exist_ok=True)
    page=(ROOT/'index.html').as_uri()
    top=run(page,'report_top',1200)
    assert 'raw105 · 17/49' in top and '21/25' in top and '候选' in top
    harness=WORK/'controls_harness.html'
    harness.write_text('''<!doctype html><meta charset="utf-8"><style>body{margin:0}iframe{width:100vw;height:100vh;border:0}#out{position:fixed;top:0;right:0;background:white;z-index:9999;font:12px monospace;max-width:420px}</style><iframe id="viewer" src="URL"></iframe><pre id="out">RUNNING</pre><script>
const tests=[];const pause=ms=>new Promise(r=>setTimeout(r,ms));function check(name,value){tests.push({name,passed:!!value});if(!value)throw Error(name);}
document.getElementById('viewer').onload=async()=>{try{const w=document.getElementById('viewer').contentWindow,d=w.document,$=id=>d.getElementById(id);await pause(400);
for(const state of ['keep','delete'])for(const raw of [89,94,105,137]){$('allState').value=state;$('allState').dispatchEvent(new w.Event('change'));$('allRaw').value=raw;$('allRaw').dispatchEvent(new w.Event('input'));await pause(160);check(state+' raw'+raw,['allObserved','allBase','allNew'].every(id=>$(id).complete&&$(id).naturalWidth===480&&$(id).naturalHeight===320)&&$('allNew').getAttribute('src').endsWith('/'+String(raw).padStart(3,'0')+'/'+state+'.png'));}
$('allRaw').value=105;$('allRaw').dispatchEvent(new w.Event('input'));d.documentElement.style.scrollBehavior='auto';w.scrollTo({top:$('allRaw').closest('section').getBoundingClientRect().top+w.scrollY,behavior:'instant'});await pause(200);document.body.dataset.qa='PASS';}catch(e){document.body.dataset.qa='FAIL';tests.push({error:String(e)});}document.getElementById('out').textContent=JSON.stringify({status:document.body.dataset.qa,tests});};</script>'''.replace('URL',page),encoding='utf-8')
    dom=run(harness.as_uri(),'report_controls',5000)
    match=re.search(r'<pre id="out">(.*?)</pre>',dom,re.S)
    record=json.loads(html.unescape(match.group(1)))
    assert record['status']=='PASS',record
    record.update(browser='Installed Google Chrome headless',top_screenshot=str(WORK/'report_top.png'),
                  control_screenshot=str(WORK/'report_controls.png'),native_size=[480,320],no_visible_browser_opened=True)
    (ROOT/'evidence/report_browser_qa.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,ensure_ascii=True))


if __name__=='__main__':
    main()
