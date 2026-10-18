"""Publish one explicitly verified node and current progress; no training edits."""
from pathlib import Path
import argparse,datetime,json,os,subprocess
W=Path(__file__).parent;P=W.parent/'r15_closure_run/publish_repo';O=W.parents[1]/'outputs/baseline_engineering_20261018'
D=Path('docs/autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/baseline_engineering_20261018');S=Path('scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/baseline_engineering_20261018');paths=[]
def target(p):return Path('\\\\?\\'+str((P/p).resolve())) if os.name=='nt' else P/p
def cp(src,p):
 q=target(p);q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(src.read_bytes());paths.append(p.as_posix())
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--model',required=True,choices=['A','B']);parser.add_argument('--step',type=int,required=True);args=parser.parse_args()
 name=f'parallel_{args.model}_{args.step:06d}_durability_receipt.json';r=json.loads((O/name).read_text());assert r['model']==args.model and r['step']==args.step and r['status']=='PASSED_ACTUAL_NAMED_PARALLEL_CHECKPOINT_CPU_READONLY'
 assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=P).strip()
 for n in ['index.html','direct_plan.json','direct_validation.json','live_direct_instance_status.json','live_B_native_status.json','remaining_disk_forecast.json',name]:
  cp(O/n,D/n)
  if n=='index.html':
   p=target(D/n);p.write_text(p.read_text(encoding='utf-8').replace('../identity_comparison_20261018/','../identity_20261018/'),encoding='utf-8')
 for n in ['audit_parallel_checkpoint_node.py','build_direct_report.py','publish_direct_plan.py','publish_checkpoint_milestone.py']:cp(W/n,S/n)
 a=json.loads((O/'live_direct_instance_status.json').read_text());b=json.loads((O/'live_B_native_status.json').read_text());disk=json.loads((O/'remaining_disk_forecast.json').read_text())
 line=f'''实际{args.model} {args.step:,}完整保存节点已CPU只读核验，凭据{name}。所有模型/Adam张量有限，CPU RNG可解析、CUDA RNG存在，采样状态={args.step}；B节点另核对SGCM和49视图评估complete。没有GPU恢复或额外训练更新，不代表资产质量通过。最新报告快照A{a['step']}/30000、B{b['iteration']}/30000，均正常并行；guardian79622和共享准入未变。剩余存储预测空闲{disk['current_free_bytes']/1e9:.2f}GB，按配置cap加10%拓扑余量估需{disk['estimated_remaining_required_bytes']/1e9:.2f}GB、余{disk['estimated_headroom_bytes']/1e9:.2f}GB；非严格点数上界，历史资产全部保留。raw129静态车形的表示/约束漏洞诊断继续有效，A/B未改目标，最终完整画面质量仍待评估。'''
 rel=Path('docs/RESEARCH_STATUS.md');p=target(rel);text=p.read_text(encoding='utf-8');i=text.index('五身份分支同步视频已交付：');text=text[:i]+line+'\n\n'+text[i:];p.write_text(text,encoding='utf-8');paths.append(rel.as_posix())
 rel=Path('docs/research_failures/entries/V82-F06.md');p=target(rel);text=p.read_text(encoding='utf-8');heading=f'## 2026-10-19：并行{args.model} {args.step//1000}K完整节点只读核验';assert heading not in text
 p.write_text(text+'\n\n'+heading+'\n\n'+line+'\n\nhuman_verdict=null；无新增训练臂、消融或checkpoint删除。\n',encoding='utf-8');paths.append(rel.as_posix())
 subprocess.run(['python','scripts/build_research_failure_index.py'],cwd=P,check=True,capture_output=True)
 paths.extend(['docs/research_failures/ENTRIES.md','docs/research_failures/IDS.md','docs/research_failures/VERSIONS.md','docs/research_failures/ids/V82.md','docs/research_failures/versions/V82-1.md']);unique=sorted(set(paths))
 spec=W/f'checkpoint_{args.model}_{args.step}_stage.nul';spec.write_bytes(b'\0'.join(p.encode() for p in unique)+b'\0');subprocess.run(['git','-c','core.safecrlf=false','add','--pathspec-from-file='+str(spec.resolve()),'--pathspec-file-nul'],cwd=P,check=True)
 staged=subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=P).decode().strip('\0').split('\0');assert set(staged)<=set(unique)
 subprocess.run(['git','diff','--cached','--check'],cwd=P,check=True);subprocess.run(['git','diff','--cached','--stat'],cwd=P,check=True)
 (W/f'checkpoint_{args.model}_{args.step}_publish_paths.json').write_text(json.dumps(unique,indent=2))
if __name__=='__main__':main()
