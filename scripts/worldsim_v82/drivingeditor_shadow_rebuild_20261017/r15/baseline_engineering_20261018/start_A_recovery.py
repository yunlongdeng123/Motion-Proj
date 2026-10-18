from pathlib import Path
import os,json,subprocess,datetime
E=Path(__file__).resolve().parent;p=E/'A_recovery_launch.json';assert not p.exists()
for pid in [5153,6099,39526]:assert not Path('/proc/'+str(pid)).exists()
auth=E/'A_replay_authorization.json'
with auth.open('x') as f:json.dump(dict(source='direct human reply: 允许，但是你得查清楚，以免后续又有这种中断。并且每隔5k保存一次',authorized_replay_updates=5846,logical_target_step=30000,resume_step=20000,checkpoint_interval=5000,from_scratch_restart=False),f,ensure_ascii=False,indent=2)
with (E/'A_recovery_controller.log').open('x') as f:child=subprocess.Popen(['/root/autodl-tmp/envs/drivingeditor/bin/python','-B','-u',str(E/'run_A_recovery_pipeline.py')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
row=dict(pid=child.pid,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),same_A_pipeline_recovery=True,authorized_extra_replay=5846,checkpoint_interval=5000)
with p.open('x') as f:json.dump(row,f,indent=2)
print(json.dumps(row))
