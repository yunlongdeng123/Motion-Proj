"""Publish actual storage recovery without touching running AutoDL jobs."""
from pathlib import Path
import json,subprocess,re
import publish_checkpoint_milestone as p
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=p.P).strip()
r=json.loads((p.O/'runtime_io_recovery_receipt.json').read_text(encoding='utf-8'))
assert r['status']=='OBSERVED_STORAGE_RECOVERY_WITH_CONTINUED_A_PROCESS' and r['agent_replay_updates']==0
a=json.loads((p.O/'live_direct_instance_status.json').read_text(encoding='utf-8'));b=json.loads((p.O/'live_B_native_status.json').read_text(encoding='utf-8'))
assert a['step']>=r['first_recovered_A_step'] and a['state']=='running' and b['status']=='complete' and b['iteration']==30000
for name in ['index.html','direct_plan.json','direct_validation.json','runtime_io_health.json','runtime_io_recovery_receipt.json','live_direct_instance_status.json','live_B_native_status.json','remaining_disk_forecast.json']:
 p.cp(p.O/name,p.D/name)
 if name=='index.html':
  dest=p.target(p.D/name);dest.write_text(dest.read_text(encoding='utf-8').replace('../identity_comparison_20261018/','../identity_20261018/'),encoding='utf-8')
for name in ['collect_runtime_io_health.py','record_io_recovery.py','build_direct_report.py','publish_direct_plan.py','publish_io_recovery.py']:
 p.cp(p.W/name,p.S/name)
p.cp(p.W/'report_phase_validation.json',p.D/'report_phase_validation.json')
d=r['refreshed_forecast']
note=f'''存储等待已实际解除：最后一次确认受阻为{r['last_confirmed_blocked_at']}，首次确认恢复为{r['first_confirmed_recovered_at']}，具体平台恢复时刻和根因未获确认。A同一PID79624、同一命令行从20,400推进至21,800，本轮最新已到{a['step']}/30000；md0由write-pending恢复active/clean，GPU重新计算，损失与梯度记录正常。没有杀进程、重启、重放、新增训练臂或改变目标；B30K与随检查点49视图评估完成保持，最终A/B删除评估仍待guardian按原队列串行执行。阻塞的磁盘预测helper252862已退出，旧23:15结果不当当前预测；{d['at']}另外重新读取空闲{d['current_free_bytes']/1e9:.3f}GB、cap加10%预测剩余{d['estimated_remaining_required_bytes']/1e9:.3f}GB、余{d['estimated_headroom_bytes']/1e9:.3f}GB，非严格上界。当前A完整已验保存节点仍5/10/15/20K，下一25/30K按实际落盘核验。完整画面报告已改回训练中并显示实际恢复证据，完成和质量门槛未放宽；没有执行关机。'''
rel=Path('docs/RESEARCH_STATUS.md');dest=p.target(rel);body=dest.read_text(encoding='utf-8')
body,count=re.subn(r'2026-10-18T23:20:33\.292178\+00:00只读检查发现A实例状态停在20,400/30000[^\n]*',lambda m:note,body,count=1);assert count==1
dest.write_text(body,encoding='utf-8');p.paths.append(rel.as_posix())
rel=Path('docs/research_failures/entries/V82-F06.md');dest=p.target(rel);body=dest.read_text(encoding='utf-8');heading='## 2026-10-19：宿主机存储恢复，A原进程继续';assert heading not in body
dest.write_text(body+'\n\n'+heading+'\n\n'+note+'\n\nhuman_verdict=null；原检查点和所有资产保留。\n',encoding='utf-8');p.paths.append(rel.as_posix())
subprocess.run(['python','scripts/build_research_failure_index.py'],cwd=p.P,check=True,capture_output=True)
p.paths.extend(['docs/research_failures/ENTRIES.md','docs/research_failures/IDS.md','docs/research_failures/VERSIONS.md','docs/research_failures/ids/V82.md','docs/research_failures/versions/V82-1.md'])
paths=sorted(set(p.paths));spec=p.W/'io_recovery_stage.nul';spec.write_bytes(b'\0'.join(x.encode() for x in paths)+b'\0')
subprocess.run(['git','-c','core.safecrlf=false','add','--pathspec-from-file='+str(spec.resolve()),'--pathspec-file-nul'],cwd=p.P,check=True)
staged=subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=p.P).decode().strip('\0').split('\0');assert set(staged)<=set(paths)
subprocess.run(['git','diff','--cached','--check'],cwd=p.P,check=True);subprocess.run(['git','diff','--cached','--stat'],cwd=p.P,check=True)
(p.W/'io_recovery_publish_paths.json').write_text(json.dumps(paths,indent=2))
