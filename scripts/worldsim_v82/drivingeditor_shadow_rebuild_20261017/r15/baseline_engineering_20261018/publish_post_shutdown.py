"""Append actual shutdown observations locally after the delivered run ends."""
from pathlib import Path
import json,subprocess
import publish_checkpoint_milestone as p

def main():
 r=json.loads((p.O/'shutdown_status_current.json').read_text())
 assert r['status']=='SHUTDOWN_REQUESTED_SSH_UNREACHABLE_OBSERVED' and r['cloud_billing_verified'] is False
 assert sum(not x['SSH_reachable'] for x in r['observations'][-3:])==3
 assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=p.P).strip()
 text=('全部A/B训练、最终零更新49帧删除评估、一次素材下载、完整画面实际复核和Git交付后，已按授权执行AutoDL关机。'
  f"实际请求{r['request_started_at']}，SSH观察完成{r['completed_at']}；入口/usr/bin/shutdown，执行前sync。"
  f"关机前交付提交{r['delivery_git_commit']}已推送并核对remote HEAD。"
  '当前A/B-aware preflight核对guardian79622、A79624、B79625、旧磁盘helper252862/252851及全部campaign worker/待启动controller均无活动，GPU计算进程为空，最终完整检查点和归档保留。'
  'SSH连续三次连接失败；这是实际SSH不可达观察，不声称已核实AutoDL平台计费。A/B干净街景质量仍未通过；没有再训练或重放，human_verdict=null。')
 for name in ['index.html','direct_plan.json','direct_validation.json','shutdown_preflight_current.json','shutdown_status_current.json']:p.cp(p.O/name,p.D/name)
 p.cp(p.W/'publish_post_shutdown.py',p.S/'publish_post_shutdown.py')
 rel=Path('docs/RESEARCH_STATUS.md');dest=p.target(rel);old=dest.read_text(encoding='utf-8')
 pending='当前交付后按授权执行全任务preflight/sync/AutoDL关机并记录SSH关闭观察，不声称核实云计费。'
 assert pending in old
 dest.write_text(old.replace(pending,text,1),encoding='utf-8');p.paths.append(rel.as_posix())
 rel=Path('docs/research_failures/entries/V82-F06.md');dest=p.target(rel);old=dest.read_text(encoding='utf-8')
 heading='## 2026-10-19：最终交付后授权AutoDL关机';assert heading not in old
 link='../../autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/baseline_engineering_20261018/'
 dest.write_text(old.rstrip()+'\n\n'+heading+'\n\n'+text+'\n\n[实际preflight]('+link+'shutdown_preflight_current.json) · [实际时间和SSH观察]('+link+'shutdown_status_current.json)。\n',encoding='utf-8');p.paths.append(rel.as_posix())
 subprocess.run(['python','scripts/build_research_failure_index.py'],cwd=p.P,check=True,capture_output=True)
 (p.W/'direct_publish_paths.json').write_text(json.dumps(sorted(set(p.paths)),indent=2))
 print(json.dumps(dict(scoped_files=len(set(p.paths)),shutdown_observed=True,training_updates=0)))
if __name__=='__main__':main()
