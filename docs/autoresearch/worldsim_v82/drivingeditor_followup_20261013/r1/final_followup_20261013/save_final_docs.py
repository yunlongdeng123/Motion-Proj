"""Archive completed run evidence and update the single current research snapshot."""
from pathlib import Path
import datetime,json,shutil
W=Path('C:/Users/dengyunlong/Documents/Codex/2026-10-12/w');O=W/'outputs';T=W/'work/drivingeditor_resume/heartbeat_20261013_1056'
R=W/'work/drivingeditor_resume/repo_active'
rel='docs/autoresearch/worldsim_v82/drivingeditor_followup_20261013/r1/final_followup_20261013'
A=R/rel;A.mkdir(parents=True,exist_ok=True)
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def copy(p,dst):
    long_dst=Path('\\\\?\\'+str(dst.resolve()))
    long_dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2('\\\\?\\'+str(p.resolve()),str(long_dst))
summary=load(O/'DrivingEditor_final_summary.json');review=load(O/'drivingeditor_red_GN11_30k/engineering-visual-review.json')
assert summary['total_training_steps']==180000 and review['review_complete'] is True and review['human_verdict'] is None
for sub in ['red_nodes','red_terminal']:
    for p in (T/sub).rglob('*'):
        if p.is_file() and p.suffix in {'.json','.jsonl','.log','.py','.js','.svg'}:
            dst=A/sub/p.relative_to(T/sub);copy(p,dst)
for p in T.glob('*'):
    if p.is_file() and p.suffix in {'.json','.py','.md'}:copy(p,A/p.name)
(A/'red_report_metadata').mkdir(exist_ok=True)
for p in (O/'drivingeditor_red_GN11_30k').glob('*.json'):copy(p,A/'red_report_metadata'/p.name)
for p in (O/'drivingeditor_red_GN11_30k/terminal_evidence').rglob('*'):
    if p.is_file():
        dst=A/'red_report_metadata/terminal_evidence'/p.relative_to(O/'drivingeditor_red_GN11_30k/terminal_evidence');copy(p,dst)
copy(O/'DrivingEditor_final_summary.json',A/'final_summary.json')
text=(O/'drivingeditor_red_GN11_30k/results-report.md').read_text(encoding='utf-8')
prefix='C:/Users/dengyunlong/Documents/Codex/2026-10-12/w/outputs/drivingeditor_red_GN11_30k/'
text=text.replace(prefix,'../autoresearch/worldsim_v82/drivingeditor_followup_20261013/r1/final_followup_20261013/red_report_metadata/')
text=text.replace('../autoresearch/worldsim_v82/drivingeditor_followup_20261013/r1/final_followup_20261013/red_report_metadata/architecture-components.svg','DRIVINGEDITOR_RED_GN11_30K_COMPONENTS.svg')
(R/'docs/v82/DRIVINGEDITOR_RED_GN11_30K_RESULTS.md').write_text(text,encoding='utf-8')
copy(O/'drivingeditor_red_GN11_30k/architecture-components.svg',R/'docs/v82/DRIVINGEDITOR_RED_GN11_30K_COMPONENTS.svg')
now=datetime.datetime.now(datetime.timezone.utc);canonical=R/'docs/v82/DRIVINGEDITOR_FOLLOWUP_20261013.json';d=load(canonical)
d['updated_utc']=now.isoformat();d['last_updated_singapore']=now.astimezone(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
d['state']='six_30k_runs_complete_reports_reviewed_saved_power_pending'
d['last_hourly_matrix_observation']=load(T/'final_matrix_inspect.json')
d['last_hourly_followup_observation']=load(T/'final_followup_inspect.json')
d['followup']['status']='both_new_scenes_complete_30000_each'
d['followup']['new_total_training_iterations_completed']=60000
d['closeout_failure_ledger_delta']='none'
d['final_closeout']={'status':summary['status'],'training_steps':180000,'checkpoint_bundles_checked':18,'human_GN11_verdict':None,
    'restoration_verified':False,'3090_initial_screen_passed':True,'target_3090_measured':False,'observed_peak_with_preprocessing_GiB':summary['including_preprocessing_observed_max_GiB'],
    'outputs':['outputs/DrivingEditor_final_report.html','outputs/DrivingEditor_final_reports.zip','outputs/drivingeditor_red_GN11_30k/index.html'],
    'evidence':rel,'old_preparation_receipts_are_historical':True,'power_signal_sent':False}
for spec in d['followup']['scene_specs']:
    sid=spec['scene_name'];isred=sid.startswith('scene0029');folder='drivingeditor_red_GN11_30k' if isred else 'drivingeditor_GMC_GN11_30k';s=load(O/folder/'summary.json')
    spec.update(terminal_iteration=30000,terminal_run_complete=True,terminal_process_exited=True,training_allowed_now=False,
        training_gates_pending=[],restoration_verified=False,human_verdict=None,saved_HTML='outputs/'+folder+'/index.html',
        engineering_visual_review_status='complete_all49_observed_and_all49_fixed_pose_times',readiness_status='REAL_TRAINING_COMPLETE_30000',training_ready=True)
    old=spec['execution']
    old.update(status='complete',terminal_iteration=30000,checkpoints_saved_at_confirmation=[10000,20000,30000],
               source_status_snapshot=rel+'/red_report_metadata/terminal_evidence/status.json' if isred else old['source_status_snapshot'])
    if 'latest_resource_snapshot' in old:old['historical_1001_resource_snapshot']=old.pop('latest_resource_snapshot')
    old['latest_confirmation']={'iteration':30000,'status':'complete','PID_exited':True,'all3node_contract_pass':True,'wall_minutes':s['wall_minutes'],
        'allocated_peak_GiB':s['allocated_peak_GiB'],'reserved_peak_GiB':s['reserved_peak_GiB'],'process_sampled_peak_GiB':s['process_peak_GiB'],
        'device_sampled_peak_GiB':s['device_peak_GiB'],'final_static_points':s['static_points'],'final_dynamic_points':s['dynamic_points'],
        'is_terminal_peak':True,'certified_3090_fit':False,'restoration_verified':False}
    spec['terminal_results']={'full_composite_psnr_db':s['full_composite_psnr_db'],'actor_composite_psnr_db':s['actor_composite_psnr_db'],
        'actor_valid_frame_count':s['valid_actor_frame_count'],'empty_actor_frame_count':s['empty_actor_frame_count'],'static_psnr_db':s['background_static_psnr_db']}
write(canonical,d)
status='''# 当前研究状态

更新：2026-10-13（新加坡）。分支 `research/worldsim-v8.2-drivingeditor-debug`。当前任务 `WS-V82-DRIVINGEDITOR-GN11-GATE-OTHER-SCENES-20261013/r1` 的训练、评价和实际分层/固定时间复核已全部完成：原四臂各30k、两个追加scene各GN11 30k，共180,000步。六个训练worker、数据/分割/SAM及CPU收集controller均已退出；18个10k/20k/30k四文件bundle、49 FP32/PNG和固定49时间的stat/header合同通过。未实际恢复完整模型/Adam/RNG，`restoration_verified=false`。最后保存、推送与已授权关机收口中，当前尚未发出电源信号。

GN11−C00全49等权PSNR：全图+0.3254dB、车体−0.1287dB、非天空车外静态+0.3175dB。完整逐步配对、全49分层、连续raw20–46的27个不利车体帧及全部固定49时间实际复核，未触发用户确认的均值下降超过1dB或明显持续相对视觉回退规则。局部车体亮度/细节/边界代价及两臂静态道路时变残影保留。真实gate仍passed，human仅C00=用户符合预期；GN11/两个新scene仍null。[四臂结果与组件图](v82/DRIVINGEDITOR_FOUR_ARM_30K_RESULTS.md)。

GMC scene0008 raw23–71，Corolla153保留静态、DEVA77/82合一个动态Gaussian；GN11完整30k，84.449分钟，全图34.308dB、车体29.698dB（39/49有效）、实际静态35.582dB。末raw62–71十个空body保留/null。实际复核保留局部接地阴影不足、静态道路黑残影和固定pose0008033时间变化。[GMC结果与组件图](v82/DRIVINGEDITOR_GMC_GN11_30K_RESULTS.md)。

红车scene0029 raw69–117，真实DEVA130、raw98/99/100选经审查的原SAM模型候选，无GT造mask或填洞；29,725真实prior点，原600k静态seed在strict0.1m排除后初始597,734点。GN11完整30k，82.611分钟，最终静态1,030,480/动态1,084,958点，全图37.524dB、车体31.151dB（32/49有效）、实际静态36.970dB；raw101–117十七个空body保留/null。全49分层及固定pose0029079的49时间实际复核：静态道路黑色车底/轮胎残影保留，固定时间中黑条块移动，后段局部绿线/浅粉道路斑块。不存在该scene的C00对照，不认证SGCM/NPS因果回退或收益。[红车结果与组件图](v82/DRIVINGEDITOR_RED_GN11_30K_RESULTS.md) · [最终证据](autoresearch/worldsim_v82/drivingeditor_followup_20261013/r1/final_followup_20261013/)。

本聊天`outputs/DrivingEditor_final_report.html`汇总并链接三个真实深色viewer；`outputs/DrivingEditor_final_reports.zip`离线保存三报告及2,646 PNG，另有原四臂和GMC单独包。所有49帧、共享ROI、raw alpha、固定相机时间控制与坏例保留；端点、空body、播放/暂停和图片加载真实浏览器QA通过。质量指标来自训练观测重建，无heldout或删除后GT。作者两scene展示身份逐帧确认，实际训练窗口未知，本地统一49扩展；不称完整论文效果复现。

六组训练全程覆盖增密、10k/20k/30k检查点、LPIPS和静态时间控制，采样无错误；完整训练最大观测6.902GiB。包括真实sky/DEVA/SAM的最高观测10.279GiB，满足3090 24GiB减2GiB容量初筛。采样是下界；分割子进程外部Torch allocator不可读（SAM内部已有峰），梯度诊断无NV样本。不认证3090容量、速度或稳定性，目标机仍须实测。每臂与阶段allocated/reserved、PID/device采样、点数和速度见报告和原memory_summary。

机制保持2024活跃代码：完整raw body opacity100、静态RGB1、寿命/速度约束与LiDAR深度衰减。SGCM修正支持/克隆谱系是中心投影及temporal-alpha代理，非准确遮挡贡献；NPS保留sum/L2/.01，六节点梯度finite，static-feature/RGB断连ratio占位0无效，不能认定绝对静态。body-only无完整shadow，M-detector未成为实际预处理；完整模型恢复、独立测试与人工质量通过未认证。

SSH仅`wm-vgpu-1011`。原矩阵根`/root/autodl-tmp/drivingeditor_repro/campaign/hybrid30k_20261012`；后续根`/root/autodl-tmp/drivingeditor_repro/campaign/followup_scene_prep_20261013`；GPU解释器`/root/autodl-tmp/envs/drivingeditor/bin/python`。完整模型、原FP32、输入和失败资产保留AutoDL数据盘，HTML/Git离线包不是大检查点备份。源身份搜索已关闭，不重置5GiB预算，不追加arms/seeds/loss网格。[机器可读状态](v82/DRIVINGEDITOR_FOLLOWUP_20261013.json)。

用户已明确授权两个新scene训练/评价/报告收口、保存推送后AutoDL关机，无需重新询问。最后重新核对vendor `/usr/bin/shutdown`和supervisord精确PID/cmdline、零owned GPU/CPU/controllers及无关任务；省略vendor的Trash删除，仅执行日志/TERM后端。当前心跳待收口后暂停，其他旧自动化保持暂停。SSH断开不等于独立平台电源证明；平台不可读时仅报告“已发出关机请求”。

本轮真实mask queue首次因Python3.9无os.pidfd_open在模型前停止，兼容修复后真实数据完整完成：[V82-F03](research_failures/entries/V82-F03.md)。最后CPU collector首轮路径断言失败，partial保留，第二轮使用事件真实metrics目录通过全部复算；不是科学否定。首次C00初始化停滞`attempts/initialization_1`及所有旧资产保留。`failure_ledger_refs=[V82-F01,V82-F02,V82-F03,V76-F03]`，任务累计delta=V82-F03，最终收口新增delta=none。子agent默认`gpt-6.1-sol/max`；无关dirty `scripts/run_worldsim_v72_e3_render_bridge.py`保留。
'''
(R/'docs/RESEARCH_STATUS.md').write_text(status,encoding='utf-8')
ep=R/'docs/EXPERIMENTS.md';e=ep.read_text(encoding='utf-8');lines=e.splitlines()
for i,line in enumerate(lines):
    if line.startswith('| WS-V82-DRIVINGEDITOR-GN11-GATE-OTHER-SCENES-20261013 / r1 |'):
        lines[i]='| WS-V82-DRIVINGEDITOR-GN11-GATE-OTHER-SCENES-20261013 / r1 | 原四臂passed gate；GMC与红车GN11均完整30k/三节点/49帧深色页与实际分层、固定时间复核完成；总180k，仅两追加场景、无同scene C00。保留shadow/静态时变/固定时间局部伪影；18节点stat/header通过但完整恢复未测；3090仅容量初筛 | [红车结果与组件图](v82/DRIVINGEDITOR_RED_GN11_30K_RESULTS.md) · [GMC结果](v82/DRIVINGEDITOR_GMC_GN11_30K_RESULTS.md) · [规则和终态](v82/DRIVINGEDITOR_FOLLOWUP_20261013.json) · [最终证据](autoresearch/worldsim_v82/drivingeditor_followup_20261013/r1/final_followup_20261013/) |'
ep.write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'archive':str(A),'state':d['state'],'total_training_steps':180000}))
