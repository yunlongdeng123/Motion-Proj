"""只接收本次已有结果；不启动模型、不设置定时任务。"""
from pathlib import Path
import subprocess,json,time,tarfile
S=Path(__file__).resolve().parent;O=S.parents[1]/'outputs/v77-nine-full';T='/root/autodl-tmp/runs/worldsim_v77/WS-V77-NINE-FULL-20260928/r1';beg=time.time()
assert not O.exists(),'新审核目录必须不存在，禁止覆盖旧结果'
while True:
    assert time.time()-beg<14400
    p=subprocess.run(['ssh','wm-3090-0811',f'cat {T}/finish_state.json'],capture_output=True,text=True)
    if p.returncode==0:
        state=json.loads(p.stdout)
        if state['state']=='failed_engineering':raise RuntimeError(state)
        if state['state']=='complete':break
    time.sleep(30)
archive=S/'review_bundle.tar';subprocess.run(['scp',f'wm-3090-0811:{T}/review_bundle.tar',str(archive)],check=True)
O.mkdir(parents=True)
with tarfile.open(archive) as tar:tar.extractall(O,filter='data')
(O/'receive_state.json').write_text(json.dumps(dict(state='media_received_not_yet_html_reviewed',remote=T,bytes=archive.stat().st_size),indent=2));print('NINE_RECEIVED',O,flush=True)
