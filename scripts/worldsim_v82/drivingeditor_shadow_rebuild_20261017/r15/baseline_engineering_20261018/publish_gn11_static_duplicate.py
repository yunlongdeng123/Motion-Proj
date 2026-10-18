"""Scoped static/dynamic separation diagnosis; preserve active training code."""
from pathlib import Path
import json,os,subprocess,datetime
W=Path(__file__).parent;P=W.parent/'r15_closure_run/publish_repo';O=W.parents[1]/'outputs/baseline_engineering_20261018'
D=Path('docs/autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/baseline_engineering_20261018');S=Path('scripts/worldsim_v82/drivingeditor_shadow_rebuild_20261017/r15/baseline_engineering_20261018');paths=[]
def target(rel):return Path('\\\\?\\'+str((P/rel).resolve())) if os.name=='nt' else P/rel
def cp(src,rel):
 p=target(rel);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(src.read_bytes());paths.append(rel.as_posix())
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=P).strip()
for p in (O/'gn11_static_duplicate_20261019').rglob('*'):
 if p.is_file():cp(p,D/'gn11_static_duplicate_20261019'/p.relative_to(O/'gn11_static_duplicate_20261019'))
for n in ['index.html','direct_plan.json','direct_validation.json','live_direct_instance_status.json','live_B_native_status.json']:
 cp(O/n,D/n)
 if n=='index.html':
  q=target(D/n);q.write_text(q.read_text(encoding='utf-8').replace('../identity_comparison_20261018/','../identity_20261018/'),encoding='utf-8')
for n in ['index.html','validation.json']:cp(O/'road_observability_20261019'/n,D/'road_observability_20261019'/n)
for n in ['audit_gn11_static_duplicate.py','audit_gn11_static_motion_features.py','build_gn11_static_duplicate_report.py','validate_gn11_static_duplicate.py','build_direct_report.py','build_road_observability_report.py','publish_direct_plan.py','publish_gn11_static_duplicate.py']:cp(W/n,S/n)
body='''用户指出raw129新五身份GN11第一阶段静态仍有a6/a37。实际CPU只读复核49帧canonical mask解码和保存的30K mask与五身份标签并集逐像素一致；raw129两车动态alpha均值0.989055/0.994077，动态body RGB误差3.62/1.96（255域）。原生静态renderer与上游归档字节一致，已实际解码主播放器第40帧并与raw129原生静态PNG核对，RGB均误差2.51/255，非动态/静态混用。标签一致不代表SAM mask是像素GT。

实际loss函数在保存的真实FP32图和mask上CPU autograd：body区域静态RGB输出梯度严格0；动态alpha损失与独立static alpha无计算连接。独立静态alpha近1也可能是正常道路，不能通过压空2D静态alpha来“修”干净街景。尚未归因每个车形像素的Gaussian贡献/深度。

更关键的是原生静态表示仍用raw_velocity做SHM位置变化、时间高斯显隐，速度正则和NPS却读取经temporal_width指数衰减的feature velocity。实际1988336静态点中1195277点49时间采样中心跨度>0.1m且feature norm<1e-6；全场景跨度中位0.372m、95分位1.324m、最大4.252m。速度阈值为模型单位/模型时间，不是m/s；全局计数不等于车体owner。已保存的固定相机49时间检查显示静态RGB随时间变，现只用CPU导出完整49帧视频，无新GPU渲染。相关3getter AST与固定上游560d57c3一致，是继承的表示/约束漏洞，非本轮源码公式误改。

初始化0.1m动态种子去重重放：实际149072动态种子、585943初始静态；raw129同时满足真实3D框与body mask的a6/a37测量点135/1215，仅0/3个在初始静态有1mm邻居。五锚帧仅0—3个，不支持大量原车测量点原封漏入静态是主因，不能据此证明所有初始化正确。

修订上一轮“动态已承担主体、主要是道路没恢复”的结论：道路观测数据仍成立，但不能由此证明车体已彻底分离。没有因果消融，不对观测不足与静态重复解释强行排主次。当前静态车形对干净街景目标不通过。B同样继承这些路径，扩大20身份不自动解决。A/B按用户原授权预算继续，未改在训模型/损失、未增加训练臂/消融；原45K暂停。human_verdict=null。
'''
rel=Path('docs/RESEARCH_STATUS.md');p=target(rel);text=p.read_text(encoding='utf-8');i=text.index('五身份分支同步视频已交付：');text=text[:i]+body+'\n[raw129实际梯度、源码与完整画面](autoresearch/worldsim_v82/drivingeditor_shadow_rebuild_20261017/v3_backend_restore_r15/baseline_engineering_20261018/gn11_static_duplicate_20261019/index.html)。\n\n'+text[i:];p.write_text(text,encoding='utf-8');paths.append(rel.as_posix())
rel=Path('docs/research_failures/entries/V82-F06.md');p=target(rel);text=p.read_text(encoding='utf-8');heading='## 2026-10-19：raw129静态重复解释与运动特征约束脱节';assert heading not in text
p.write_text(text+'\n\n'+heading+'\n\n'+body,encoding='utf-8');paths.append(rel.as_posix())
subprocess.run(['python','scripts/build_research_failure_index.py'],cwd=P,check=True,capture_output=True)
paths+=['docs/research_failures/ENTRIES.md','docs/research_failures/IDS.md','docs/research_failures/VERSIONS.md','docs/research_failures/ids/V82.md','docs/research_failures/versions/V82-1.md'];paths=sorted(set(paths))
ignored=subprocess.run(['git','check-ignore','--stdin','-z'],cwd=P,input=b'\0'.join(p.encode() for p in paths)+b'\0',capture_output=True).stdout.decode().strip('\0').split('\0');ignored=[p for p in ignored if p];regular=[p for p in paths if p not in ignored]
for subset,force,name in [(regular,False,'regular'),(ignored,True,'ignored')]:
 if not subset:continue
 spec=W/f'duplicate_stage_{name}.nul';spec.write_bytes(b'\0'.join(p.encode() for p in subset)+b'\0')
 subprocess.run(['git','-c','core.safecrlf=false','add']+(['-f'] if force else [])+['--pathspec-from-file='+str(spec.resolve()),'--pathspec-file-nul'],cwd=P,check=True)
staged=subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=P).decode().strip('\0').split('\0');assert set(staged)<=set(paths)
subprocess.run(['git','diff','--cached','--check'],cwd=P,check=True);subprocess.run(['git','diff','--cached','--stat'],cwd=P,check=True)
(W/'duplicate_publish_paths.json').write_text(json.dumps(paths,indent=2))
