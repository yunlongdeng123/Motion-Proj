"""远端局部归档、更新唯一状态快照、同步可复用模板；不提交其他工作。"""
from pathlib import Path
import tarfile,json,shutil,datetime,subprocess
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028')
NEW=ROOT/'unknown_detection_v3';REC=NEW/'r2_recovery_20261029'
REPO=Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
DOC=REPO/'docs/autoresearch/v43_ownership_20261028'
STAMP=datetime.datetime.now(datetime.timezone.utc).isoformat()
BACKUP=REC/'publication_history'/STAMP.replace(':','')
files=[];templates=[]
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO).strip(), 'Preserve unexpected staged work'
def put(path,data):
    if path.exists():
        rel=path.relative_to(REPO) if path.is_relative_to(REPO) else Path('active_template')/path.name
        b=BACKUP/rel;b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,b)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.suffix in ['.py','.js','.json','.md','.html']:data=data.replace(b'\r\n',b'\n')
    path.write_bytes(data)
    if path.is_relative_to(REPO):files.append(str(path.relative_to(REPO)))
with tarfile.open(REC/'publication.tar.gz') as ar:
    for m in ar.getmembers():
        dst=(REPO/m.name).resolve()
        assert m.isfile() and not m.issym() and dst.is_relative_to(REPO)
        data=ar.extractfile(m).read()
        if dst==DOC/'r2_recovery_20261029/index.html':
            local=b'file:///C:/Users/dengyunlong/Documents/Codex/2026-10-22/ni-x-a/outputs/'
            data=data.replace(b'../../../GS_Scene_Diagnostics_20261027/',local+b'GS_Scene_Diagnostics_20261027/')
            data=data.replace(b'../bounded_white_v1/index.html',local+b'V43_Ownership_20261027/gpu_v43/bounded_white_v1/index.html')
        put(dst,data)
status='''# 当前研究状态

当前任务：WS-V43-R2-RECOVERY-20261029。用户指出旧检测面板 pending 和后续结果回退。本轮恢复原 R2 继承链，完成第一场景 raw77—125 共49帧；无新增训练。旧检测面板实际仅3帧，已折叠为历史记录。此前147帧空初始化方法改变了种子、彩噪查询及框外基线，不能视为 R2 的逐帧推广，保留为未采纳消融。

CPU先按原算法补齐49帧接触区，三锚点掩码完全一致；GPU重放原三锚点，行集合与 G/alpha/局部白底浮点差均为0，再按原分类规则生成49帧 existing。原R2同帧existing种子、body/contact/浮点越界查询、8次评估上限与局部范围外existing基线全部保留；原生删除并重渲染49帧，逐帧重新计算贡献。五项继承门槛及独立完整浮点审核通过，441张PNG已取回。完整逐像素浮点留远端；本地含全部PNG、删行ID、范围和审计。

共享工作台场景1默认 v43_r2_inherited49，49帧主滑块同步实拍/existing/R2白底及删行3D；旧R2三帧留作对照。当前49无数据缺帧。新增unknown中心框外0，既有种子框外单列；白底是 G+(1-alpha) 的局部空洞诊断，框外使用existing，不是补景成功或全局删除无影响。

AI视觉核查49张局部图及11帧原尺寸，未发现严重坏帧或明显跳变；部分边缘仍有灰暗和彩边，raw101/113/114较明显。没有语义GT和P/R结论，quality_accepted=false、human_verdict=null。当前研究方向仍是车辆相关unknown检测，不能把白底本身作为语义正确证明。

第二、三场景既有10K车身和编辑保留；本轮没有重做其R2继承。后续先由用户验收场景1逐帧，再以相同输入合同复现实例2/3，禁止空种子静默回退。边缘混合支撑和关联阴影仍待核验；不扩大框外unknown。

记录：[R2继承回归报告](autoresearch/v43_ownership_20261028/r2_recovery_20261029/index.html)。执行开销：CPU接触区96.32秒，49帧GPU扩展79.84秒；锚点重放和种子生成另记日志，无训练更新。源码和报告归档并推送后，按用户要求检查活动进程并关机；旧队列不恢复。failure_ledger_refs=[V82-F01]；failure_ledger_delta=none。
'''
put(REPO/'docs/RESEARCH_STATUS.md',status.encode())
p=REPO/'docs/EXPERIMENTS.md';s=p.read_text()
row='| WS-V43-R2-RECOVERY-20261029 / inherited-r1 | 场景1真实R2三锚点逐值复现、49帧原生重放与同帧3D；零训练，独立审计通过，边缘残留待人工验收 | [报告与组件图](autoresearch/v43_ownership_20261028/r2_recovery_20261029/index.html) |'
if 'WS-V43-R2-RECOVERY-20261029' not in s:s=s.replace('|---|---|---|','|---|---|---|\n'+row,1)
put(p,s.encode())
p=DOC/'STATE.json';state=json.loads(p.read_text())
state.update(checked_utc=STAMP,current_task='WS-V43-R2-RECOVERY-20261029')
state['r2_recovery']=dict(phase='delivered_pending_human_review',context='14705303724557273004_3105_000_3125_000',raws=list(range(77,126)),method='inherited_R2',training_updates=0,all_5_gates_passed=True,anchor_float_max_difference=0,full49_mechanical_audit_passed=True,complete_PNG_row_ID_delivery=True,full_float_local_copy=False,quality_accepted=False,human_verdict=None,ai_visual_review='minor_boundary_residue_no_severe_failure',old_queue_resumed=False,remote_root=str(REC/'run_r1'),report='r2_recovery_20261029/index.html',scene2_3_r2_inherited_this_turn=False)
if 'temporal_multiscene' in state:state['temporal_multiscene'].update(accepted_as_R2_successor=False,phase='preserved_unadopted_empty_seed_ablation',reason='seed/query/outside baseline differ from original R2')
if 'target_detection' in state:state['target_detection'].update(latest_report='r2_recovery_20261029/index.html',phase='scene1_inherited_R2_49_user_review')
put(p,json.dumps(state,ensure_ascii=False,indent=2).encode())
code=REPO/'scripts/worldsim_v82/v43_ownership_20261028';active=ROOT.parent/'persistent_world_20261027/code/diagnostic_templates/v43_semantic_gpu_v2'
mapping={
 'build_scene_workspaces.py':code/'diagnostic_template/build_scene_workspaces.py',
 'scene_target_detection.js':code/'diagnostic_template/scene_target_detection.js',
 'install_temporal_ui.py':code/'unknown_detection_v3/temporal_multiscene_20261029/install_temporal_ui.py'}
for name in ['patch_ui.py','install_r2_inherited_ui.py','qa_r2_inherited_ui.js','qa_ui.js']:
    mapping[name]=code/'unknown_detection_v3/r2_recovery_20261029'/name
for name,source in mapping.items():
    dest=active/name;put(dest,source.read_bytes());assert dest.read_bytes()==source.read_bytes();templates.append(str(dest))
subprocess.run(['git','add','--']+files,cwd=REPO,check=True)
subprocess.run(['git','diff','--cached','--check'],cwd=REPO,check=True)
subprocess.run(['git','diff','--cached','--stat'],cwd=REPO,check=True)
(REC/'publication_receipt.json').write_text(json.dumps(dict(files=files,templates=templates,backup=str(BACKUP),updated_utc=STAMP),indent=2))
print('STAGED',len(files),'TEMPLATES',len(templates))
