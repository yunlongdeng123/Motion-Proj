import pathlib,subprocess,json
root=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-PIPELINE-R2-20260926/r1')
p=root/'controller_state.json'
if p.exists():
    s=json.loads(p.read_text())
    if s['state']=='running':raise RuntimeError('inspect running controller before relaunch')
with (root/'controller.log').open('w') as log:
    script=next(p for p in [pathlib.Path(__file__).with_name('r2_controller.py'),pathlib.Path(__file__).with_name('v77_r2_controller.py')] if p.is_file())
    proc=subprocess.Popen(['/root/autodl-tmp/envs/motionproj/bin/python',str(script)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'pid':proc.pid,'run':str(root)}))
