"""停止退化试验后的最终归档；不运行模型、不修改原FULL资产。"""
import datetime, json, pathlib, re, shutil, subprocess

REPO=pathlib.Path('/root/autodl-tmp/motion_proj_v77')
ROOT=pathlib.Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-DELETE-REPAIR-20260927/r1')
HERE=pathlib.Path(__file__).parent
EVIDENCE=REPO/'docs/autoresearch/worldsim_v77/delete_repair_20260927'
REVIEW=ROOT/'review'
BACKUP=ROOT/'repo_backups'/('closeout_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))

def load(p): return json.loads(p.read_text())
def dump(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def write(rel,text):
    dst=REPO/rel
    if dst.exists():
        bak=BACKUP/rel;bak.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dst,bak)
    dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(text)

state=load(ROOT/'drive_state.json')
assert state['state']=='stopped_regression'
assert not any(x['candidate'] for x in load(ROOT/'background_admission.json')['scenes'])
assert load(REVIEW/'page_validation.json')['missing']==[]
config=load(REPO/'configs/worldsim_v77/delete_pipeline_current.json')
assert config['active_task']=='WS-V77-DELETE-FULL-20260927' and not config['experimental_precise_mask_route_enabled']

write('docs/RESEARCH_STATUS.md','''# 当前研究状态

更新：2026-09-27，v77。`WS-V77-DELETE-REPAIR-20260927/r1`出现明确退化，已按用户“差则记录failure并回退”的要求停止，恢复上一轮DriveEditor完整DELETE方案作为默认。没有正在运行的生成任务；不自动重启该精确轮廓输入实验。

## 当前默认

`configs/worldsim_v77/delete_pipeline_current.json`指向`WS-V77-DELETE-FULL-20260927/r1`：scene_0230/actor22、scene_0255/actor25，原DriveEditor mask配置、冻结Ω背景、原GLB。旧BUILD/QUERY入口与1a2d5189相同，原资产未替换。回退保留旧版再生车辆、模糊与Ω错误遮挡的已知缺陷，不表示高保真通过。V7.6关闭状态不变。

## 本轮范围与结果入口

两旧scene A/B各30帧；新增official_000/actor12 A30帧、B10帧后停止，共17个完成窗口，0230零证据所以B复用A。三例新补景均出现白车或灰白车形，检测器能抓清晰车但漏检残影；全部阻断进入Ω。第三目标没有旧DriveEditor/GLB基线，保留为失败诊断，未纳入默认世界。零新Ω前向、零训练，旧资产与全部对照保留。

审核页：`C:/Users/dengyunlong/Documents/Codex/2026-09-26/xia/outputs/v77-delete-repair/index.html`。上方为恢复后的旧版原视频/factual/DELETE，下方为三案例失败对照、精确mask、真实证据、原生输出和guard。详情见[最终报告](v77/DELETE_REPAIR.md)；唯一run证据在`/root/autodl-tmp/runs/worldsim_v77/WS-V77-DELETE-REPAIR-20260927/r1`。

## 下一步与边界

等待用户看报告；不继续相同失败配置、不扩训练集、不恢复ProPainter、MOVE、定时任务或历史关机安排。若继续，本轮只留下“生成条件范围与最终精确写回范围可单独控制”的未验证假设，不预填改善。GT/LiDAR辅助、开发scene、隐藏背景无GT、官方训练重叠未知等边界保持。

`failure_ledger_refs: [V77-F02]`；`failure_ledger_delta: updated V77-F02`；`human_verdict: null`。
''')

report=(REVIEW/'report.md').read_text()
report=report.replace('(index.html)','(../autoresearch/worldsim_v77/delete_repair_20260927/review_link.md)')
report=re.sub(r'\]\(([a-z_]+\.json)\)',r'](../autoresearch/worldsim_v77/delete_repair_20260927/\1)',report)
write('docs/v77/DELETE_REPAIR.md',report)

failure='''
## 三场景精确mask试验退化，按用户要求回退

`WS-V77-DELETE-REPAIR-20260927/r1`以SAM2精确轮廓/小膨胀替换删除矩形，配合原RGB、冻结Ω深度、GT相机与LiDAR支持做证据warp；A精确mask直接DriveEditor，B真实证据优先且只生成残余洞，同seed42/25步/1024×576/10帧窗口。0230/CAM5/f18–47、0255/CAM3/f65–94各A/B30帧；新增official_000/actor12/CAM0的A30帧、B10帧。17窗口完成后因明确视觉反例及用户回退要求停止，未补齐第三例。0230 B因零证据复用A。旧/新初始化历史不同，非严格单变量因果比较；本轮A/B为固定窗口对照。

0230删除范围减少39.92%，0255减少82.96%，但0230生成清晰白车，0255与第三例产生灰白车形/残留轮廓；原生输出已出现，不能归因GLB或后续Ω。背景来源仅接受原RGB，需贴地包络排除、20cm LiDAR支持、双时刻深度/RGB一致；30帧准备中的有效覆盖为0%、1.69%、0.109%，不足以检验充分真实背景条件能否消除幻觉。稀疏/严格规则也可能造成低覆盖，不能证明未被观测。

GroundingDINO-tiny+SAM2在原视频目标正控制中全部命中；对生成图，0230 A/B各29/30帧拦截，0255 A0/30、B3/30，第三例各0/10。灰白残影会漏检，零检测不能通过；三例结合助手视觉反例全部不给后续Ω提供路径。零新Ω前向、原GLB不改，人工verdict null。邻车操作范围更准不等于语义完全保真；没有把局部mask改善当DELETE成功。

用户授权失败回退后，默认配置指回`WS-V77-DELETE-FULL-20260927/r1`，旧BUILD/QUERY与1a2d5189无差异，所有资产与失败输出完整保留。旧版已知再生车/模糊/错误遮挡仍存在；第三目标没有旧基线，只作为失败诊断保存。停止该精确轮廓直接生成入口，不再机械换seed或阈值。官方get_blank/删除分支使用矩形，细轮廓是输入分布变化事实，未证明为唯一原因；生成条件与精确写回分离仍是未执行假设。不据此宣布整个background+asset路线或模型家族失败。

8项语义/准入测试通过，90帧来源准备和已生成图像素合同验证通过；34视频/820帧实解码、71页面引用与JS语法检查通过，未冒称浏览器交互已测。首窗漏带已有串行CFG导致OOM零输出，恢复后运行，单独保留工程错误日志。

见[最终报告与architecture](../../v77/DELETE_REPAIR.md)、[实际帧数](../../autoresearch/worldsim_v77/delete_repair_20260927/comparison_plan.json)、[失败拦截](../../autoresearch/worldsim_v77/delete_repair_20260927/background_admission.json)、[回退验证](../../autoresearch/worldsim_v77/delete_repair_20260927/rollback_validation.json)。`failure_ledger_delta: updated V77-F02`；`human_verdict: null`。
'''
fp=REPO/'docs/research_failures/entries/V77-F02.md'
old=fp.read_text()
assert '## 三场景精确mask试验退化，按用户要求回退' not in old
write(fp.relative_to(REPO),old+failure)
ep=REPO/'docs/EXPERIMENTS.md';old=ep.read_text()
new=re.sub(r'^- `WS-V77-DELETE-REPAIR-20260927/r1`：.*$', '- `WS-V77-DELETE-REPAIR-20260927/r1`：三场景入口试验失败并回退；实际数量与边界见[报告](v77/DELETE_REPAIR.md)、[登记](autoresearch/worldsim_v77/delete_repair_20260927/registration.json)、[收口](autoresearch/worldsim_v77/delete_repair_20260927/closeout.json)。', old, flags=re.M)
assert old!=new
write(ep.relative_to(REPO),new)

names=['summary','mask_summary','evidence_summary','guard_summary','comparison_plan','drive_state','validation','background_admission','observations','rollback','rollback_validation','track_audit','page_validation']
for name in names:
    src=REVIEW/(name+'.json')
    write(EVIDENCE.relative_to(REPO)/(name+'.json'),src.read_text())
write(EVIDENCE.relative_to(REPO)/'review_link.md','''# 本地审核入口

`C:/Users/dengyunlong/Documents/Codex/2026-09-26/xia/outputs/v77-delete-repair/index.html`

回退后的旧版两scene三栏视频与本轮三scene失败对照。依赖同级`v77-delete-full`目录的既有视频，勿单独移动HTML。完整远端证据在`/root/autodl-tmp/runs/worldsim_v77/WS-V77-DELETE-REPAIR-20260927/r1`。
''')
source_names=['repair_common','repair_prepare','repair_masks','repair_evidence','repair_drive','repair_guard','repair_guard_smoke','repair_admission','repair_validate','repair_package','repair_track_audit','repair_stop','repair_rollback_report','repair_rollback_review','repair_rollback_closeout','test_repair']
for name in source_names:
    write(pathlib.Path('scripts/worldsim_v77')/(name+'.py'),(HERE/(name+'.py')).read_text())

closeout={'task_id':'WS-V77-DELETE-REPAIR-20260927','run_id':'r1','state':'stopped_regression_rolled_back','completed_driveeditor_windows':len(state['completed']),'actual_outputs':state['actual_outputs'],'new_omega_forwards':0,'baseline_task':config['active_task'],'baseline_run':'r1','baseline_implementation_commit':'1a2d5189','failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'updated V77-F02: precise masks regress; evidence insufficient; vehicle guard misses blobs; reverted default','human_verdict':None,'unit_tests':8,'diagnostic_video_decode':load(REVIEW/'summary.json')['total_decoded_frames'],'page_validation':load(REVIEW/'page_validation.json'),'backup_root':str(BACKUP)}
dump(ROOT/'closeout.json',closeout);write(EVIDENCE.relative_to(REPO)/'closeout.json',json.dumps(closeout,ensure_ascii=False,indent=2)+'\n')
subprocess.run(['/root/autodl-tmp/envs/motionproj/bin/python','scripts/build_research_failure_index.py'],cwd=REPO,check=True)
print(json.dumps({'state':closeout['state'],'backup':str(BACKUP)},ensure_ascii=False))
