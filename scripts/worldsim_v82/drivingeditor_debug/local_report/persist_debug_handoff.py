import datetime,json,tomllib
from pathlib import Path
HERE=Path(__file__).resolve().parent
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
m=json.loads((HERE/'task_manifest.json').read_text(encoding='utf-8-sig'))
m.update(updated_utc=now,gpu_owner='vgpu_component_diagnostic_solmax',gpu_owner_scope='root-approved controlled original-math30k-to30300 training only',
    gpu_status_path='continuation_diagnostic/original30k_to30300/status.json',
    current_experiment='original30k_to30300',next_experiment='opacity_outside single-variable candidate; CPU preparation/review only until explicit root GPU handoff',
    component_inference={'status':'complete','exit_code':0,'case_count':6,'camera_count':5,'baseline40_png_parity':'exact','protected64_files':'unchanged'},
    cpu_audits={'loss':'loss_audit/handoff_summary.json','SGCM':'sgcm_data_audit/audit_findings.json'},
    component_interpretation='20k static environment contribution directly contains road confetti; 30k static reset exposes it. Environment-only swap does not restore30k; static20k swap does.',
    acceptance_not_yet_met=True,paper_complete_reproduction_claimed=False)
(HERE/'task_manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
registry=Path('C:/Users/dengyunlong/.codex/automations/drivingeditor-50k/automation.toml')
saved=tomllib.loads(registry.read_text(encoding='utf-8-sig'))
marker='\n\n最新已核验排错交接（2026-10-12）'
prompt=saved['prompt'].split(marker)[0]+marker+'''：
任务manifest在BASE/campaign/vgpu_bugfix_20261012/task_manifest.json；仅silver场景debug，other_scene_queue_pause.json及next_scene30k_20261012/user_pause.json已落盘，不再推进其它场景。
6组件组合×5固定相机推理已exit0。20/30k基线40PNG精确匹配，各64项原文件hash/mtime不变。均值5帧composite：20k35.3261dB，30k10.8526；30k只换env20k11.7777，只换static20k33.548。30k白化主要跟静态reset相关；20k的static_env_contribution已经是道路彩噪，static_foreground颜色很暗，不能把死clone直接当彩噪来源。固定光线训练/导出composite/static/alpha/depth一致；dynamic共享env差值公式残差0。详细component_diagnostic/{diagnostic_manifest,cpu_audit,train_eval_semantics,no_modification_sha}.json，图片contact_sheets与component_breakdown_raw10。
CPU审计：loss_audit/handoff_summary.json和HANDOFF.md；sgcm_data_audit/audit_findings.json和handoff.txt。确认LiDAR同像素last-write顺序依赖，抽查8–12%有效深度像素不是最近点。确认当前实际目标函数存在逐像素分配歧义：static空/dynamic全图与proper split同极低损失（不是有限Gaussian全视角可达性证明）。NPS值不爆炸，dynamic位置LR正确。静态仍SHM和时间alpha，不是严格time-invariant；论文基础PVG/静态Gs说明有歧义。20k约999266额外exactXYZ复制，最大簇158570；773564 clone带点在连续[-.48,.48]内alpha上界低于原kernel1/255，因此没有像素贡献，不能产生彩噪。初始dynamic只有2.18%地面点，不是初始大面积道路分入dynamic。raw17地面2293样本全部有其它帧真实观测/LiDAR支持，不能以道路从未观测笼统解释。
候选全部隔离未部署：loss_audit/candidates/nearest_depth.patch、opacity_outside.patch（15项CPU测试local/remotePASS）；sgcm_data_audit/candidate/paper_sgcm_visibility_guard.patch默认关闭（7项CPU PASS）；local reset_candidate/stop_late_static_reset.patch（60项CPU schedule check）。它们分别是几何投影修正、未核实论文背景域的单变量假说、SGCM未报告资格选择、late-reset调度修正，不能混在一次实验或宣称已完整复现论文。
当前唯一GPU任务original30k_to30300，输出BASE/campaign/vgpu_bugfix_20261012/continuation_diagnostic/original30k_to30300。launcher/status/log/exit在component_diagnostic/continuation_plan/continuation_launcher_status.json、continuation_training.log、continuation_launcher.exit、continuation_worker.pid；2026-10-11T18:46:52Z初始launcher31936/worker31939是历史PID，必须实时核对。root_coordination_receipt已批准此唯一原数学300步对照；prepared_plan SHAf12d6c0801997d0a09020617c7629f5f41abfab4083ebbf0400f1e715be6d205，harness ac9745d5c0cbab7701ae8be5beec9587d57ea79288a3167259d0af1d0e164458，observer f60faca00ec0d5b10b80d14d3e8ca07125ada5bff06d1f0fb8c43c5fec6d42d1。不得重复启动。完整原30k四件/RNG/sampler/alloptimizer严格恢复，horizon50000，只有model_path/start_checkpoint账务变化；实际import源/kernel校验，额外eval恢复全部RNG；早期5帧、末30300全49帧，保存terminal_30300四件及独立diagnostic_snapshot_manifest，不是新的formalcheckpoint事件。原30001 TB loss=.2980644106864929/PSNR8.590572357177734；比较实际delta，不偷偷放宽阈值。训练已进入30001..，其它GPU不得并发。
下一步CPU准备中的outside-opacity单变量由paper_loss_bug_audit_solmax负责、sgcm_data_bug_audit_solmax独立review；需读实际handoff/代码diff/config/sourceSHA，baseline完成释放GPU后root审核才能采用。它仅增加mask外软alpha=0惩罚λ=.05（保留论文前景Eq13），是诊断假说，不是作者已确认项。必须同30k初始状态、同30001..30300样本和原50k调度、全49末eval，与原版比较composite质量、static在mask外可观测区质量、dynamic mask外alpha与支持；不同时修改depth/SGCM/reset。若未来扩展步数，需明确新实验窗口和无reset混杂，不把不同训练步数的结果冒充同条件。即使图像改善也不能直接宣布完整论文复现。
深色报告outputs/drivingeditor_vgpu_debug/index.html已生成含架构图、reset曲线、实际分支图/环境分解、6组组件表、深度/SGCM/静态时间/域歧义及证据。当前为排查中，不是完成；local refs21项通过，图像/曲线已查看，浏览器播放未验证。报告持续更新，保持全部旧3090original主对比不变。'''
payload={'mode':'update','id':saved['id'],'kind':'heartbeat','name':'DrivingEditor vGPU 论文版排错','status':'ACTIVE','rrule':saved['rrule'],'targetThreadId':saved['target_thread_id'],'prompt':prompt}
if 'notification_policy' in saved:payload['notificationPolicy']=saved['notification_policy']
(HERE/'automation_vgpu_debug.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(payload,ensure_ascii=True))
