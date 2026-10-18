"""CPU-only collector. Never starts/restarts a GPU worker or controller."""
from pathlib import Path
import subprocess,time,json,tarfile,shutil,sys
W=Path(__file__).parent;O=W.parents[1]/'outputs/identity_comparison_20261018'
R='/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1'
T=R+'/identity_completion_20261018';PY='/root/autodl-tmp/envs/drivingeditor/bin/python'
def ssh(cmd):return subprocess.check_output(['ssh','-o','ConnectTimeout=12','wm-vgpu-1011',cmd],text=True,encoding='utf-8')
def write(v):(W/'branch_collection_status.json').write_text(json.dumps(v,indent=2),encoding='utf-8')
write(dict(state='WAITING_ACTUAL_FROZEN_BRANCHES'))
last=''
while True:
    try:
        value=ssh(f"test -f {T}/branch_queue_status.json && cat {T}/branch_queue_status.json || true").strip()
        if value:
            d=json.loads(value)
            if value!=last:print(value,flush=True);last=value
        ready=ssh(f"test -f {T}/branches_old5/branch_receipt.json && test -f {T}/branches_r15/branch_receipt.json && echo ready || true").strip()
        if ready=='ready':break
        q=json.loads(ssh(f'cat {R}/shadow_rebuild_20261017_v3/backend_restore_r15/repair_4k_20261018/queue_status.json'))
        if q['state']=='NEEDS_ATTENTION_ASSETS_PRESERVED':raise RuntimeError(q)
        time.sleep(30)
    except subprocess.CalledProcessError:
        write(dict(state='SSH_TEMPORARILY_UNAVAILABLE_RETRYING'));time.sleep(30)
meta=json.loads(ssh(f'{PY} {T}/code/pack_branches.py'))
write(dict(state='DOWNLOADING',archive=meta));local=W/Path(meta['path']).name
subprocess.run(['scp','wm-vgpu-1011:'+meta['path'],str(local)],check=True)
assert local.stat().st_size==meta['bytes']
dest=W/'branch_download';dest.mkdir(exist_ok=True)
with tarfile.open(local) as tar:
    assert all(m.isfile() and not m.issym() and not m.islnk() and (dest/m.name).resolve().is_relative_to(dest.resolve()) for m in tar.getmembers())
    tar.extractall(dest)
for n in meta['models']:shutil.copytree(dest/('branches_'+n),W/'branches'/('branches_'+n),dirs_exist_ok=True)
if (dest/'renders_repair4k').is_dir():shutil.copytree(dest/'renders_repair4k',O/'assets/repair4k',dirs_exist_ok=True)
write(dict(state='ENCODING_SYNCHRONIZED_PLAYER',models=meta['models']))
subprocess.run([sys.executable,str(W/'build_sync_player.py')],check=True)
subprocess.run([sys.executable,str(W/'build_sync_exports.py')],check=True)
write(dict(state='FROZEN_BRANCH_PLAYER_COMPLETE',models=meta['models'],all_inputs_are_actual_renders=True))
print('Frozen branch player is complete',flush=True)
