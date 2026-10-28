"""本轮可复现源码、验收记录和当前状态；不暂存其他用户改动。"""
from pathlib import Path
import tarfile,datetime,json,shutil,subprocess
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028')
NEW=ROOT/'unknown_detection_v3';REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
DOC=REPO/'docs/autoresearch/v43_ownership_20261028'
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat();backup=NEW/'publication_history'/(stamp.replace(':','')+'_temporal');files=[]
def put(path,data):
 if path.exists():
  rel=path.relative_to(REPO) if path.is_relative_to(REPO) else Path('active_template')/path.name
  b=backup/rel;b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,b)
 path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data.replace(b'\r\n',b'\n'))
 if path.is_relative_to(REPO):files.append(str(path.relative_to(REPO)))
with tarfile.open(NEW/'temporal_publication.tar.gz') as ar:
 for m in ar.getmembers():
  dst=(REPO/m.name).resolve();assert m.isfile() and not m.issym() and dst.is_relative_to(REPO);put(dst,ar.extractfile(m).read())
text='''# 当前研究状态

当前任务：WS-V43-TEMPORAL-MULTISCENE-20261029。用户认可旧 bounded-white 第三列三帧作为当前视觉基线，授权第一场景逐帧检验并将第二、三场景同流程接入共享工作台，现已开启GPU。此前第三场景停止范围已被本次授权覆盖，旧训练/自动队列未恢复。

已复用第二、三场景暂停前完成的10K模型及原生motion，不重复训练；各场景49帧结构、4类×31帧编辑已取回并接入。原生与既有ProPainter生成通道分开，默认原生RGB。新增三场景各49帧冻结unknown诊断，训练更新0。新臂从每帧空候选开始，只读原始body和统一车底query，蓝框形状归一化传递；同raw删除后重算完整alpha*T。旧R2依赖的场景一专属种子未冒充跨场景证据。

147帧完整浮点审核通过：独立重置、三场景参数一致、范围内查询贡献无未选中有效候选、框外新增中心0、局部范围外像素差0、G_after+T×白底方程一致。局部合成保护框外像素不等于全局3D删除。原始完整浮点现场保留远端，本地取回全部PNG、行ID、bool范围及远端完整审核；旧三帧本地完整浮点对照保留。

质量不能写成全部通过：新body/contact三锚点AI目视接近旧R2，但contact alpha更高；扩大整条局部带只多删背景，没有改善三锚点contact alpha，故未扩跑。场景3 raw13仍有车底紫褐带，raw37/49/61邻车边缘存在粘连观感。重点时序检查段：场景1 raw104–105/108–109、场景2 raw149–153、场景3 raw21–25。行ID变化不等价于RGB闪烁，没有语义GT或P/R结论。

同一共享入口：outputs/GS_Scene_Diagnostics_20261027/index.html。新完整对照：outputs/V43_Ownership_20261027/gpu_v43/temporal_multiscene_20261029/index.html。详细机器证据和组件图见[TEMPORAL_MULTISCENE_20261029.html](autoresearch/v43_ownership_20261028/TEMPORAL_MULTISCENE_20261029.html)。

下一步等待用户逐帧视觉验收，再针对跨边界混合支撑、车底残留和邻车边界做有约束改进；不继续扩大框外unknown，不把白洞当补景成功。正式渲染前有六结果真实pilot；两次接口名称不一致各额外raw98单帧已保留并单列，修正后完整重放，没有训练恢复或预算变更。

GPU/发布/本地取回和关闭凭据见STATE.json及本轮delivery记录；关机只在全部后续依赖、Git推送及活动任务检查结束后执行。failure_ledger_refs=[V82-F01]；failure_ledger_delta=none。新147帧human_verdict=null，quality_accepted=false。
'''
put(REPO/'docs/RESEARCH_STATUS.md',text.encode())
p=REPO/'docs/EXPERIMENTS.md';text=p.read_text();row='| WS-V43-TEMPORAL-MULTISCENE-20261029 / source-body-contact-r1 | 三场景147帧独立局部剥离；2/3场景10K原生编辑接入；冻结训练0，机械审计通过、视觉待验且保留badcase | [组件与记录](autoresearch/v43_ownership_20261028/TEMPORAL_MULTISCENE_20261029.html) |'
if 'WS-V43-TEMPORAL-MULTISCENE-20261029' not in text:text=text.replace('|---|---|---|','|---|---|---|\n'+row,1)
put(p,text.encode())
p=DOC/'STATE.json';state=json.loads(p.read_text());state['checked_utc']=stamp;state['current_task']='WS-V43-TEMPORAL-MULTISCENE-20261029'
state['temporal_multiscene']=dict(phase='published_pending_human_review',contexts=3,frames_per_context=49,training_updates=0,mechanical_audit_passed=True,quality_accepted=False,human_verdict=None,old_queue_resumed=False,third_scene_held=False,remote_root=str(NEW/'temporal_multiscene_20261029_r1'),report='TEMPORAL_MULTISCENE_20261029.html',full_float_local_copy=False,complete_PNG_and_row_ID_delivery=True)
if 'target_detection' in state:state['target_detection'].update(third_scene_held=False,latest_report='TEMPORAL_MULTISCENE_20261029.html',phase='multiscene_user_review',next='Inspect complete temporal sequence; investigate cross-boundary support without outside relabeling')
put(p,json.dumps(state,ensure_ascii=False,indent=2).encode())
code=REPO/'scripts/worldsim_v82/v43_ownership_20261028/unknown_detection_v3/temporal_multiscene_20261029'
active=ROOT.parent/'persistent_world_20261027/code/diagnostic_templates/v43_semantic_gpu_v2'
for name in ['install_temporal_ui.py','scope.py','render_temporal.py','temporal_audit.py','audit_delivery.py']:
 put(active/name,(code/name).read_bytes())
put(active/'patch_shared_scene_entries.py',(code/'patch_shared_scene_entries.py').read_bytes())
subprocess.run(['git','add','--']+files,cwd=REPO,check=True);subprocess.run(['git','diff','--cached','--check'],cwd=REPO,check=True)
subprocess.run(['git','diff','--cached','--stat'],cwd=REPO,check=True)
(NEW/'temporal_publication_files.json').write_text(json.dumps(dict(files=files,backup=str(backup),active_template=str(active)),indent=2))
print('STAGED',len(files))
