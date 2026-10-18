"""Publish the observed storage wait; preserve running jobs and unrelated Git edits."""
from pathlib import Path
import json,subprocess
import publish_checkpoint_milestone as p

assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=p.P).strip()
h=json.loads((p.O/'runtime_io_health.json').read_text(encoding='utf-8'))
assert h['status']=='BLOCKED_HOST_STORAGE_WRITE_WAIT'
assert h['remote_mutations']==0 and h['training_updates']==0
for name in ['index.html','direct_plan.json','direct_validation.json','runtime_io_health.json','live_direct_instance_status.json','live_B_native_status.json']:
 p.cp(p.O/name,p.D/name)
 if name=='index.html':
  dest=p.target(p.D/name);dest.write_text(dest.read_text(encoding='utf-8').replace('../identity_comparison_20261018/','../identity_20261018/'),encoding='utf-8')
for name in ['collect_runtime_io_health.py','build_direct_report.py','check_report_phase_contract.py','publish_direct_plan.py','publish_storage_wait.py']:
 p.cp(p.W/name,p.S/name)
p.cp(p.W/'report_phase_validation.json',p.D/'report_phase_validation.json')
note=f'''{h['at']}只读检查发现A实例状态停在20,400/30000，文件最后更新于{h['A_status_mtime_utc']}；主线程79624为D状态、wchan=md_write_start.part.0，md0阵列write-pending，GPU采样15104MiB已用/17128MiB空闲/利用率0%。B原生30K与随检查点49视图评估已完成，最终A/B删除评估尚未执行。20K完整检查点此前已核验，A原进程和内存状态保留，没有杀进程、重启、重放、修改训练目标或关机。独立磁盘预测任务252862也等待同一md写入路径，最新成功容量预测仍为历史22:47UTC值。证据确认宿主机存储写入等待，具体硬件故障或平台维护原因未查明，需要AutoDL平台排查；不擅自修改共享宿主机RAID。原ETA失效，继续窄只读监测同一进程能否恢复。只读采集脚本仅写本地证据和报告；不会再次启动额外磁盘预测写入。最终完整49帧交付、Git核对和无worker检查后关机的授权保持，当前不关机。'''
rel=Path('docs/RESEARCH_STATUS.md');dest=p.target(rel);body=dest.read_text(encoding='utf-8');i=body.index('五身份分支同步视频已交付：');dest.write_text(body[:i]+note+'\n\n'+body[i:],encoding='utf-8');p.paths.append(rel.as_posix())
rel=Path('docs/research_failures/entries/V82-F06.md');dest=p.target(rel);body=dest.read_text(encoding='utf-8');heading='## 2026-10-19：A20.4K宿主机存储写入等待';assert heading not in body
dest.write_text(body+'\n\n'+heading+'\n\n'+note+'\n\nhuman_verdict=null；未新增训练臂，既有20K和B30K资产保留。\n',encoding='utf-8');p.paths.append(rel.as_posix())
subprocess.run(['python','scripts/build_research_failure_index.py'],cwd=p.P,check=True,capture_output=True)
p.paths.extend(['docs/research_failures/ENTRIES.md','docs/research_failures/IDS.md','docs/research_failures/VERSIONS.md','docs/research_failures/ids/V82.md','docs/research_failures/versions/V82-1.md'])
paths=sorted(set(p.paths));spec=p.W/'storage_wait_stage.nul';spec.write_bytes(b'\0'.join(x.encode() for x in paths)+b'\0')
subprocess.run(['git','-c','core.safecrlf=false','add','--pathspec-from-file='+str(spec.resolve()),'--pathspec-file-nul'],cwd=p.P,check=True)
staged=subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=p.P).decode().strip('\0').split('\0');assert set(staged)<=set(paths)
subprocess.run(['git','diff','--cached','--check'],cwd=p.P,check=True);subprocess.run(['git','diff','--cached','--stat'],cwd=p.P,check=True)
(p.W/'storage_wait_publish_paths.json').write_text(json.dumps(paths,indent=2))
