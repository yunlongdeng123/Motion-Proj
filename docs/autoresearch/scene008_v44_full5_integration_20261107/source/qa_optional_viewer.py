"""Exercise the real central viewer in installed headless Chrome, without a server."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT.parents[1]/'work/v44_full5_viewer_qa'
CHROME = Path('C:/Program Files/Google/Chrome/Application/chrome.exe')


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    publication = json.loads((ROOT/'evidence/viewer_publication_receipt.json').read_text(encoding='utf-8'))
    index = Path(publication['viewer'])/'index.html'
    harness = WORK/'harness.html'
    content = '''<!doctype html><meta charset="utf-8"><style>body{margin:0}iframe{width:100vw;height:100vh;border:0}#out{position:fixed;top:0;right:0;background:white;z-index:9999;font:12px monospace;max-width:450px}</style><iframe id="viewer" src="URL"></iframe><pre id="out">RUNNING</pre><script>
const tests=[];const pause=ms=>new Promise(r=>setTimeout(r,ms));function check(name,value){tests.push({name,passed:!!value});if(!value)throw Error(name);}
document.getElementById('viewer').onload=async()=>{try{
const w=document.getElementById('viewer').contentWindow,d=w.document,$=id=>d.getElementById(id);await pause(900);
check('original default experiment',w.GSDATA.defaultExperiment==='v44_20261107');check('original top selection',$('arm').value==='v44_20261107');check('original fullclip selection',$('fullclipGroup').value==='5');
const info=w.GSDATA.scene.experiments.find(e=>e.id==='v44_full5_20261107');check('candidate image_only',info.viewer_scope==='image_only');check('only three RGB channels',JSON.stringify(info.channels)===JSON.stringify(['deleted_five','joint','gt']));
$('arm').value=info.id;$('arm').dispatchEvent(new w.Event('change'));await pause(1000);
check('real candidate canvas',$('renderCanvas').dataset.arm===info.id);check('geometry canvas hidden',['sceneCanvas','bevCanvas','profileCanvas'].every(id=>$(id).closest('.panel').hidden));check('geometry controls disabled',['loadCase','caseFile','rayMode','unitToPixel','sourceJump','bevMode','rank','fit3d','camera3d','regionSelect'].every(id=>!$(id)||$(id).disabled));check('non-RGB channels disabled',[...$('channel').options].every(o=>info.channels.includes(o.value)||o.disabled));check('pixel quantities explicitly unknown',$('pixelMetrics').dataset.precision==='not-exported-image-only');
for(const state of ['joint','deleted_five']){$('channel').value=state;$('channel').dispatchEvent(new w.Event('change'));await pause(300);check('canvas state '+state,$('renderCanvas').dataset.channel===state);}
const candidateOption=[...$('fullclipGroup').options].find(o=>o.textContent.includes('a5车顶恢复'));check('optional fullclip option',!!candidateOption);$('fullclipGroup').value=candidateOption.value;$('fullclipGroup').dispatchEvent(new w.Event('change'));
for(const raw of [89,94,105,137]){$('fullclipFrame').value=raw;$('fullclipFrame').dispatchEvent(new w.Event('input'));await pause(160);check('fullclip candidate raw'+raw,$('fullclipImage').complete&&$('fullclipImage').naturalWidth===480&&$('fullclipImage').naturalHeight===320&&$('fullclipImage').getAttribute('src').endsWith('/delete/'+String(raw).padStart(3,'0')+'.png'));check('actual frozen reference raw'+raw,$('fullclipBase').getAttribute('src')==='assets/v44_20261107/v44/'+String(raw).padStart(3,'0')+'.png');}
check('tradeoff visible',$('fullclipVersionNote').textContent.includes('21/25')&&$('fullclipVersionNote').textContent.includes('12/24'));
$('fullclipFrame').value=105;$('fullclipFrame').dispatchEvent(new w.Event('input'));d.documentElement.style.scrollBehavior='auto';d.body.style.scrollBehavior='auto';w.scrollTo({top:$('fullclip').getBoundingClientRect().top+w.scrollY,behavior:'instant'});await pause(200);document.body.dataset.qa='PASS';
}catch(e){document.body.dataset.qa='FAIL';tests.push({error:String(e)});}document.getElementById('out').textContent=JSON.stringify({status:document.body.dataset.qa,tests});};
</script>'''.replace('URL', index.as_uri())
    harness.write_text(content, encoding='utf-8')
    cmd = [str(CHROME), '--headless=new', '--no-sandbox', '--disable-gpu', '--allow-file-access-from-files',
           '--hide-scrollbars', '--no-first-run', '--window-size=1440,1100', '--virtual-time-budget=7000',
           '--user-data-dir='+str(WORK/'chrome_profile'), '--screenshot='+str(WORK/'candidate_fullclip.png'), '--dump-dom', harness.as_uri()]
    result = subprocess.run(cmd, capture_output=True, timeout=45, creationflags=subprocess.CREATE_NO_WINDOW)
    (WORK/'runtime_dom.html').write_text(result.stdout.decode('utf-8', errors='replace'), encoding='utf-8')
    (WORK/'chrome_stderr.txt').write_bytes(result.stderr)
    dom = result.stdout.decode('utf-8', errors='replace')
    match = re.search(r'<pre id="out">(.*?)</pre>', dom, re.S)
    assert match, 'No browser QA result'
    import html
    record = json.loads(html.unescape(match.group(1)))
    assert result.returncode==0 and record['status']=='PASS', record
    record.update(browser='Installed Google Chrome headless', real_central_index=str(index),
                  published_index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
                  screenshot=str(WORK/'candidate_fullclip.png'), dom=str(WORK/'runtime_dom.html'),
                  no_visible_browser_opened=True, no_GPU_render_or_model_load=True)
    (ROOT/'evidence/viewer_runtime_qa.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=record['status'], checks=len(record['tests']), screenshot=record['screenshot'])))


if __name__=='__main__':
    main()
