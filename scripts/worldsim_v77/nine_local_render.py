"""本次任务的有限本地渲染队列，等待九个资产各一次；不是定时任务。"""
from pathlib import Path
import subprocess,json,time,tarfile,os,sys
S=Path(__file__).resolve().parent;R=S/'render';REMOTE='/root/autodl-tmp/runs/worldsim_v77/WS-V77-NINE-FULL-20260928/r1';BLENDER='D:/software/Blender/blender.exe'
R.mkdir(exist_ok=True)
if not (R/'registration.json').exists():
    subprocess.run(['scp',f'wm-3090-0811:{REMOTE}/render_setup.tar',str(S/'render_setup.tar')],check=True)
    with tarfile.open(S/'render_setup.tar') as t:t.extractall(R,filter='data')
reg=json.loads((R/'registration.json').read_text());names=[s['name'] for s in reg['scenes']];done=[];begin=time.time()
while len(done)<len(names):
    assert time.time()-begin<14400,'本次有限等待超过4小时，停止并检查'
    for name in names:
        if name in done:continue
        dest=R/name
        if (dest/'render_done.json').exists():done.append(name);continue
        check=subprocess.run(['ssh','wm-3090-0811',f"test -f {REMOTE}/{name}/asset/paint_state.json && cat {REMOTE}/{name}/asset/paint_state.json"],capture_output=True,text=True)
        if check.returncode:continue
        state=json.loads(check.stdout)
        if state['state']=='failed_engineering':raise RuntimeError((name,state))
        if state['state']!='complete_pending_visual_review':continue
        ad=dest/'asset';ad.mkdir(exist_ok=True)
        for file in ['actor_pbr.obj','actor_pbr.mtl','actor_pbr.jpg','actor_pbr_metallic.jpg','actor_pbr_roughness.jpg','source_rgba.png','source_context.png','source_full.png','source_core.png','registration.json']:
            if not (ad/file).exists():subprocess.run(['scp',f'wm-3090-0811:{REMOTE}/{name}/asset/{file}',str(ad/file)],check=True)
        if not (dest/'actor.glb').exists():
            with (dest/'canonical.log').open('w') as log:subprocess.run([BLENDER,'--background','--python',str(S/'nine_canonical_blender.py'),'--',str(dest)],stdout=log,stderr=subprocess.STDOUT,check=True)
        if not (dest/'orientation.json').exists():subprocess.run([sys.executable,str(S/'nine_orientation.py'),'--root',str(R),'--scene',name],check=True)
        with (dest/'factual.log').open('w') as log:subprocess.run([BLENDER,'--background','--python',str(S/'nine_blender.py'),'--','--root',str(R),'--scene',name],stdout=log,stderr=subprocess.STDOUT,check=True)
        archive=dest/'render_result.tar'
        with tarfile.open(archive,'w') as tar:
            for p in [dest/'actor.glb',dest/'actor_axis_aligned.glb',dest/'canonical.json',dest/'orientation.json',dest/'orientation_contact.jpg',dest/'orientation_0',dest/'orientation_180',dest/'asset_views',dest/'actor_layers']:
                tar.add(p,arcname=str(p.relative_to(dest)))
            if (dest/'axis_error_control').exists():tar.add(dest/'axis_error_control',arcname='axis_error_control')
        subprocess.run(['scp',str(archive),f'wm-3090-0811:{REMOTE}/{name}/render_result.tar'],check=True)
        subprocess.run(['ssh','wm-3090-0811',f'tar -xf {REMOTE}/{name}/render_result.tar -C {REMOTE}/{name}'],check=True)
        summary=json.loads((dest/'actor_layers/render_summary.json').read_text());row=dict(scene=name,layers=len(summary['rows']),state='complete',human_verdict=None)
        (dest/'render_done.json').write_text(json.dumps(row,indent=2));done.append(name);(R/'local_state.json').write_text(json.dumps(dict(done=done,total=len(names)),indent=2));print('LOCAL_RENDER_DONE',row,flush=True)
    if len(done)<len(names):time.sleep(30)
print('ALL_LOCAL_RENDER_DONE',done,flush=True)
