"""Publish only task code/docs and the common template. Never stage user files."""
from pathlib import Path
import io,zipfile,base64,subprocess,json,hashlib,datetime
B=Path(__file__).resolve().parents[2];W=Path(__file__).parent;VIEW=B/'work/persistent_world_20261027';O=B/'outputs/V43_Ownership_20261027/gpu_v43'
state=json.loads((O/'remote_state.json').read_text(encoding='utf8'));report=json.loads((O/'semantic_delivery_v1/report.json').read_text());formal=[s for s in state['scenes'] if s['stages'].get('formal10k',{}).get('status',{}).get('phase')=='complete']
doc='''# 当前研究状态

当前任务：WS-V43-OWNERSHIP-STRUCTURE-20261028。沿用v42框架，先推进语义/实例归属诊断和车辆结构；再研究静态车辆表达分解及显露背景修复。当前输入为三个已完成V42 30K的开发场景，各49帧。旧队列和自动心跳未恢复，原V42 checkpoint与操作结果保留。用户本轮明确覆盖额度20%停止线，额度只记录，收到用户停止通知后保存边界。

实际GPU/进程/节点只看[当前快照](autoresearch/v43_ownership_20261028/STATE.json)。已完成19类+unknown的147帧冻结SegFormer-B0推理，以及完整Gaussian行的alpha*T可见贡献、同像素实际目标LiDAR深度支持。类别预测不是GT，掩码贡献不是正确身份。大量静态目标像素贡献位于与车身不相容的中心深度，支持背景外观污染或混合支撑的候选解释，不能全部整颗搬入车辆。

结构探针从带来源的实测目标点重建Actor局部资产，复用冻结static/environment；每case fresh10K，完整fresh300s+restore300s smoke单列。原百万行自由动态分支继续作为对照。此探针改变表示、容量、颜色、损失与RGB缩放，不能冒称完整GN11继承或把原30K与新10K直接当单因素比较；同初始化无约束10K控制隔离几何预算。还没有unknown分类器学习或背景联合优化的优越性证明。

实验合同、明确差异、验收与下一步见[本轮协议](autoresearch/v43_ownership_20261028/EXPERIMENT_CONTRACT.md)；[组件图](autoresearch/v43_ownership_20261028/architecture.svg)。每case一个离线HTML，原重建、原编辑、新结构节点与19类联合着色共用帧轴，目标橙框、其他车辆蓝框；CPU/GPU标签编码分别版本化，原owner不覆盖。

当前完整正式10K场景数：FORMAL_COUNT / 3。这是执行完成数，quality_accepted=false，human_verdict=null。v43自身物理编辑只做目标相关诊断，旧八类物理请求及其固定ProPainter结果保留；不把新增checkpoint/版本/诊断帧重复计正式案例。

固定后端仍为同一预训练ProPainter，输入/权重/参数不变；新局部资产移动/转向/删除诊断按原请求执行，原生与生成结果分开。历史阶段二规模和封存协议见[原v42计划](autoresearch/persistent_world_20261027/PLAN.md)，未运行封存编辑选方法。

代码：scripts/worldsim_v82/v43_ownership_20261028；新campaign：/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028。模型、数据、日志不入Git。保留用户桥接脚本未提交；保留旧checkpoint spillover物理依赖。failure_ledger_refs=[V82-F01]，failure_ledger_delta=none；当前只是增加同口径诊断，不新造普遍失败结论。
'''.replace('FORMAL_COUNT',str(len(formal)))
payload={}
for p in W.iterdir():
    if p.is_file() and p.suffix in ['.py','.cjs','.md']:payload['scripts/worldsim_v82/v43_ownership_20261028/'+p.name]=p.read_bytes()
payload['docs/RESEARCH_STATUS.md']=doc.encode()
prefix='docs/autoresearch/v43_ownership_20261028/'
payload[prefix+'STATE.json']=json.dumps(state,ensure_ascii=True,indent=2).encode();payload[prefix+'EXPERIMENT_CONTRACT.md']=(W/'EXPERIMENT_CONTRACT.md').read_bytes();payload[prefix+'architecture.svg']=(O/'architecture.svg').read_bytes()
template_files=['scene_viewer.js','scene_viewer.html','scene_edit_geometry.js','scene_sampling.js','scene_semantics.js','scene_workspace.css']
template={name:(VIEW/name).read_bytes() for name in template_files};template['style.css']=(B/'outputs/GS_Scene_Diagnostics_20261027/style.css').read_bytes()
template['architecture.svg']=(O/'architecture.svg').read_bytes();template['build_diagnostic_template.py']=(W/'build_diagnostic_template.py').read_bytes()
template['README.md']='''# V43 通用GPU诊断模板

每个case共用这些模板；必须另导出scene/cloud/identity/edit_geometry/foreign_clouds/semantic_observations。GPU语义放gpu_semantics/index.js及frame_*.js；深度索引可选。v43_groups.js负责已实际完成新节点，未完成时保持空文件。新旧行ID不同命名空间，不能用原30K 0/1/U覆盖新Actor局部种子；二维预测与原owner分别保存。GPU map为Cityscapes19+unknown，旧CPU map是独立版本。含目标橙框与其他原始车辆蓝框；没有RGB三维背景底图。每次build后做路径、图像和实际Canvas检查。
'''.encode()
template={k:v.replace(b'\r\n',b'\n') for k,v in template.items()}
template['manifest.json']=json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in template.items()},indent=2).encode()
for name,data in template.items():payload['scripts/worldsim_v82/v43_ownership_20261028/diagnostic_template/'+name]=data
payload={k:v.replace(b'\r\n',b'\n') for k,v in payload.items()}
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
    for name,data in payload.items():z.writestr(name,data)
script='''import io,zipfile,base64,json,hashlib,time
from pathlib import Path
repo=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42');R=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028');z=zipfile.ZipFile(io.BytesIO(base64.b64decode(DATA)));receipt={}
for name in z.namelist():
 assert name.startswith(('scripts/worldsim_v82/v43_ownership_20261028/','docs/autoresearch/v43_ownership_20261028/')) or name=='docs/RESEARCH_STATUS.md'
 p=repo/name;assert p.resolve().is_relative_to(repo.resolve());p.parent.mkdir(parents=True,exist_ok=True);content=z.read(name)
 if p.exists() and p.read_bytes()!=content:
  backup=R/'publication_history'/time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())/name;backup.parent.mkdir(parents=True,exist_ok=True);backup.write_bytes(p.read_bytes())
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_bytes(content);tmp.replace(p);assert p.read_bytes()==content;receipt[name]=hashlib.sha256(content).hexdigest()
 if '/diagnostic_template/' in name:
  target=R.parent/'persistent_world_20261027/code/diagnostic_templates/v43_semantic_gpu_v2'/p.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content);assert target.read_bytes()==content
index=repo/'docs/EXPERIMENTS.md';s=index.read_text();marker='WS-V43-OWNERSHIP-STRUCTURE-20261028'
if marker not in s:
 entry='| '+marker+' | 三开发场景49帧语义与真实Gaussian贡献、静态车身深度支持及Actor局部结构10K探针；执行与质量分开 | [当前状态](RESEARCH_STATUS.md)、[协议](autoresearch/v43_ownership_20261028/EXPERIMENT_CONTRACT.md) |\\n'
 lines=s.splitlines(True);position=next((i+1 for i,l in enumerate(lines) if l.startswith('|---') or l.startswith('| ---')),None)
 if position is None:s+='\\n'+entry
 else:lines.insert(position,entry);s=''.join(lines)
 index.write_text(s)
print(json.dumps(dict(files=len(receipt),sha256=receipt,templates_verified=True,user_bridge_untouched=True)))
'''.replace('DATA',repr(base64.b64encode(buf.getvalue()).decode()))
r=subprocess.run(['ssh','wm-vgpu-1011','python3 -'],input=script.encode(),capture_output=True);assert r.returncode==0,r.stderr.decode(errors='replace');(O/'publication_receipt.json').write_bytes(r.stdout);print(json.dumps({k:v for k,v in json.loads(r.stdout).items() if k!='sha256'}))
