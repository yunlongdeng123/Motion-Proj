import json
from pathlib import Path
import shutil
from datetime import datetime, timezone, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / 'work/drivingeditor_resume/repo_active'
PREP = ROOT / 'work/drivingeditor_instances/prepared/scene0008_two_actor_v1'
DEPLOY = ROOT / 'work/drivingeditor_instances/deployment'
OUT = ROOT / 'outputs'
EVIDENCE = REPO / 'docs/autoresearch/worldsim_v82/drivingeditor_instances_20261013/r1/cpu_ready_gpu_queued'
EVIDENCE.mkdir(parents=True, exist_ok=True)
now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8))).isoformat()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path, value): Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams['svg.fonttype'] = 'none'
fig, ax = plt.subplots(figsize=(15.5, 5.6), facecolor='#0b1220')
ax.set(xlim=(0, 1), ylim=(0, 1)); ax.set_facecolor('#0b1220'); ax.axis('off')
def box(x, y, w, h, title, subtitle, color='#19384a'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012',fc=color,ec='#5cbfb3',lw=1.3))
    ax.text(x+w/2,y+h*.69,title,ha='center',va='center',fontproperties=font,fontsize=14,color='#f0f6fc')
    ax.text(x+w/2,y+h*.30,subtitle,ha='center',va='center',fontproperties=font,fontsize=11,color='#bbd0e3')
def arrow(a,b,color='#7ccfc5'):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=16,color=color,lw=1.5,connectionstyle='arc3,rad=0'))
ax.text(.5,.96,'v82 实例编辑基线 · 只验证 GMC 双动态车辆',ha='center',fontproperties=font,fontsize=19,color='white')
box(.015,.59,.18,.23,'Waymo 训练先验','真实 3D 框 / track\nLiDAR + RGB')
box(.265,.59,.18,.23,'车辆局部高斯','G0：皮卡 / G1：Jeep\n每个点保留实例 ID')
box(.515,.59,.19,.23,'整体刚体变换','R_i(t) · x_local + t_i(t)\n同步旋转协方差 / SH 方向')
box(.775,.59,.205,.23,'全场景联合渲染','同一深度排序 / α·T\n背景和两实例共同遮挡')
box(.265,.24,.18,.22,'轨迹与编辑接口','先固定真实轨迹\n单车删除 / 局部平移 / yaw',color='#503b24')
box(.515,.24,.19,.22,'原 GN11 静态背景','只移除这两车框内中心\n停车继续静态 · 首阶段冻结')
box(.775,.24,.205,.22,'渲染与联合监督','整图 RGB / 深度\n逐实例可见身份贡献')
box(.015,.24,.18,.22,'真实 DEVA77 / 82','98 / 98 已审查 mask\n不把 3D 框画成分割标签')
arrow((.195,.705),(.265,.705)); arrow((.445,.705),(.515,.705)); arrow((.705,.705),(.775,.705))
arrow((.355,.46),(.605,.59)); arrow((.705,.35),(.775,.59)); arrow((.88,.59),(.88,.46))
ax.plot([.105,.105,.88],[.24,.17,.17],color='#8aa9dd',lw=1.5)
arrow((.88,.17),(.88,.24),color='#8aa9dd')
ax.text(.43,.13,'真实 DEVA 可见掩码监督',ha='center',fontproperties=font,fontsize=11,color='#a7c3e6')
ax.text(.5,.035,'无车辆 PVG 重复运动 · CPU 合同已通过 · GPU 测试排队，不抢其他任务',ha='center',fontproperties=font,fontsize=12,color='#e6be70')
fig.savefig(OUT/'DrivingEditor_instance_editor_components.png',dpi=160,bbox_inches='tight',facecolor=fig.get_facecolor())
fig.savefig(OUT/'DrivingEditor_instance_editor_components.svg',bbox_inches='tight',facecolor=fig.get_facecolor(),metadata={'Date':None})
plt.close(fig)
report = read(PREP/'preparation_report.json')
launch, queue = read(DEPLOY/'queue_launch.json'), read(DEPLOY/'queue_status.json')
summary = {'task_id':'WS-V82-DRIVINGEDITOR-INSTANCES-20261013','run_id':'r1','updated_sgt':now,
    'state':'CPU_INFRA_VALIDATED_GMC_GPU_SMOKE_QUEUED','scope':'GMC scene0008 raw23..71 two moving actors only',
    'latest_user_scope_source':'2026-10-13: 3个场景中只有一个多实例，用该场景测，可GPU/多线程CPU但不能挤其他任务',
    'training_priors':['genuine Waymo laser 3D boxes','raw track ID','genuine per-frame pose','real reviewed DEVA77/82 masks'],
    'representation':'static GN11 background + local rigid G0/G1, fixed GT poses initially; no vehicle PVG',
    'input':{'frames':49,'instances':2,'points':298075,'known_mask_pairs':98,'mask_pairs':98,
             'retained_static_seed_points':579855,'actual_warmstarted_static_points_after_GPU_exclusion':None,
             'tracks':[{'raw_track':'NoQttdyd4C16Y_zkGhiFKg','DEVA':77,'points':137174,'valid_frames':49,'FRONT_visible_frames':7},
                       {'raw_track':'fXVhuN-RuJDvZwnkQ-0YSQ','DEVA':82,'points':160901,'valid_frames':49,'FRONT_visible_frames':39}]},
    'validation':{'CPU_tests_passed':67,'CUDA_tests_pending':1,'local_real_GMC49_CPU_edit_contract':'PASS',
                  'target_Python39_real_GMC49_CPU_edit_contract':'PASS','GPU_training_started':False,
                  'core_CPU_state_restore_exact':True,'full_scene_GPU_model_Adam_RNG_restore_verified':False,
                  'photoreal_edit_quality_verified':False,'human_quality_verdict':None},
    'queue':{'directory':'/root/autodl-tmp/drivingeditor_repro/campaign/instance_editor_20261013/queue_gmc_r1',
             'pid':launch['pid'],'state':queue['state'],'first_GPU_test_steps':300,'max_wait_hours':24,'poll_seconds':300,
             'background_frozen':True,'pose_refinement_enabled':False,'new_SGCM_clones':False},
    'budget':{'first_GPU_compatibility_steps':300,'single_GMC_fixed_trajectory_total_step_limit':30000,
              'checkpoints':[10000,20000,30000],'no_silver_red_new_runs':True,'no_loss_seed_grid':True},
    'resource_policy':{'StreetCrafter_priority':True,'other_GPU_tasks_exclusive':True,'CPU_max_threads':4,'nice':10,
                       'kill_pause_other_work':False,'shutdown_authorized_in_new_task':False},
    'automation':{'id':'drivingeditor-30k','status':'ACTIVE','interval_minutes':60,'other_automations_changed':False},
    'paths':{'campaign':'/root/autodl-tmp/drivingeditor_repro/campaign/instance_editor_20261013',
             'spec':'/root/autodl-tmp/drivingeditor_repro/campaign/instance_editor_20261013/specs/scene0008_023_071.json',
             'source':'scripts/worldsim_v82/drivingeditor_instances','evidence':str(EVIDENCE.relative_to(REPO))},
    'failure_ledger_refs':['V82-F01','V82-F02','V82-F03'],'failure_ledger_delta':'none',
    'engineering_attempts':[{'phase':'CPU preparation before latest GMC-only scope','issue':'initial selected CPU interpreter lacked scipy',
        'resolution':'used existing DrivingEditor interpreter without environment install','failed_partial_preserved':True,'scientific_negative':False}],
    'limits':['GT box prior is engineering baseline, not new research contribution','no fair same-scene OmniRe comparison',
              'nonrigid objects unsupported','no true shadow mask or edited GT','fixed topology first, clone/prune API supported but not anisotropic split',
              'background Gaussian-center box removal can retain cross-boundary extent/shadows','shared GPU admission cannot replace real runtime validation']}
write(OUT/'DrivingEditor_instance_editor_status.json',summary)
write(REPO/'docs/v82/DRIVINGEDITOR_INSTANCE_EDITOR_20261013.json',summary)
write(EVIDENCE/'summary.json',summary)
for name in ['queue_launch.json','queue_status.json','real_GMC_CPU_contract.json','real_GMC_CPU_contract_python39.json','scene0008_023_071.json','runtime_preparation.json','selected_two_actor_bindings.json','automation_active.toml']:
    if (DEPLOY/name).exists(): shutil.copy2(DEPLOY/name,EVIDENCE/name)
for name in ['preparation_report.json','trajectories.json']:
    shutil.copy2(PREP/name,EVIDENCE/name)
for name in ['DrivingEditor_instance_editor_components.png','DrivingEditor_instance_editor_components.svg']:
    shutil.copy2(OUT/name,EVIDENCE/name)
body = '''# v82 DrivingEditor 实例编辑基础设施

当前 task/run：`WS-V82-DRIVINGEDITOR-INSTANCES-20261013/r1`。按用户最新范围，只用 GMC 的两辆动态车验证，其他停车保留原静态分支。代码、真实 CPU 输入和单实例编辑合同已准备；GPU 当前被其他训练持锁，300步兼容短测正在等待，尚无新的照片编辑质量结论。

![组件图](DrivingEditor_instance_editor_components.png)

## 表示与监督

新车辆采用 `G_s ∪ T_0(t)G_0_local ∪ T_1(t)G_1_local`。实例本身不再有 PVG 速度或时间 envelope，时间变化来自整体刚体轨迹。三维框和 track 明确作为训练先验；非刚体不属于本版。局部高斯旋转与协方差同步，SH 颜色使用局部视线方向。删除、车辆局部米制平移、局部 yaw 和场景内实例资产替换共用统一渲染入口。

二维身份贡献使用完整场景一次深度排序中的 `alpha * transmittance` 权重，包括另一辆车与静态背景的遮挡。身份通道不足时分批提交同一完整点集，绝不只渲染一辆车的孤立 alpha 来监督可见 mask。每个已知身份全 H×W 内外两域 MSE 再等权平均，系数100；未知 mask 不当作0。原静态RGB1、RGB/深度衰减、静态寿命/速度约束及原 NPS 公式保留，首阶段静态/环境参数冻结。

背景使用本 scene 原 GN11 30k 先验，按两个真实车框移除静态 Gaussian 中心。它不保证移除 Gaussian 跨框尾部或车影；移动车后补景仍需图像审查。旧 SGCM 只以背景已训练结果继承，本版不把静态点克隆直接塞入新的刚体实例；新 SGCM 迁移尚未做。首轮固定拓扑；core 有保身份及 Adam 的 clone/prune，但各向异性 split 尚未接入。

## 真实输入

| 目标 | DEVA | raw track | 真实LiDAR点 | 标注有效 / 可见 / 已知mask帧 |
|---|---:|---|---:|---:|
| GMC皮卡 | 77 | NoQttdyd4C16Y_zkGhiFKg | 137,174 | 49 / 7 / 49 |
| Jeep SUV | 82 | fXVhuN-RuJDvZwnkQ-0YSQ | 160,901 | 49 / 39 / 49 |

共298,075点，98/98真实已审查 mask。标注存在与 FRONT 可见性分开；离场空 mask 有真实审查来源，不能由缺失3D标注自动推断。原600k静态 seed 仅过滤两车框后保留579,855点；这是数据准备数量，实际继承模型经 GPU 删点后的静态点数待测。真实RGB可测颜色仅以投影/深度可见规则赋值，未测颜色显式记录 color_known=false 并零初始化；没有合成XYZ模板点。

框内点的实例归属是先验，并不保证每个框内点都是车体；二维可见监督和后续编辑坏例用于检查道路或背景污染。原掩码、3个scene旧资产与早先全车辆准备均保留，最新范围没有新增银车/红车训练。

## 已验证和待验证

本地67项CPU检查通过，1项实际CUDA检查待验。远端实际Python3.9、4CPU线程/nice10完成全49时刻真实GMC查询：删/移/转目标车，另一实例的几何、方向、外观和opacity参数严格不变；查询不写回模型，刚体 core CPU状态恢复逐元素相同，CUDA未初始化。编辑后另一车辆的可见像素可以因合法遮挡改变，这不等于其资产被修改。

CPU合同不认证完整场景模型/Adam/RNG的GPU恢复，也不认证照片重建或编辑质量。首个GPU任务300步，用于接口、梯度、资源与兼容核验；通过后先看两个实例的真实编辑，再在同一GMC固定轨迹基线上继续到总30k预算，每10k完整保存。不会增加另外两个场景、seed/loss网格或自动启用位姿细化。

目前等待controller PID20065，目录 `queue_gmc_r1`。等待不导入Torch、不持GPU锁、不建GPU worker；5分钟稀疏检查、最长24小时。其他GPU任务和StreetCrafter优先，CPU最多4线程/nice10。兼容异常不自动重跑；竞争任务新出现时worker保存priority_yield状态退出。每小时心跳已改为只跟踪本任务，无变化安静。此新阶段没有关机或终止其他任务授权。

## 运行入口

```bash
ROOT=/root/autodl-tmp/drivingeditor_repro/campaign/instance_editor_20261013
PY=/root/autodl-tmp/envs/drivingeditor/bin/python
# 默认只验证，不导入模型或启动GPU
"$PY" -B "$ROOT/infra/train.py" --spec "$ROOT/specs/scene0008_023_071.json"
# GPU测试已由queue排队，不要重复运行--run
# 真实实例checkpoint产生后，按raw track独立编辑
"$PY" -B "$ROOT/infra/render_edits.py" --spec "$ROOT/specs/scene0008_023_071.json" \
  --checkpoint ACTUAL_CHECKPOINT --track NoQttdyd4C16Y_zkGhiFKg \
  --operation translate --translate-local-m 0.5 0 0 --output NEW_DIRECTORY
```

渲染入口也默认只验证，真实导出需显式 --run 和空闲资源。平移是当前车辆局部x/y/z、单位米，yaw绕当前局部z和车体中心，考虑scene PCA旋转。替换第一版仅支持同场景已有实例资产，不宣称外部车辆资产库或重照明。新结果会接入已有统一HTML，不再增加多个HTML入口。

设计参照 [DriveStudio RigidNodes](https://github.com/ziyc/drivestudio/blob/main/models/nodes/rigid.py) 的局部实例与轨迹组织及 [get_init_objects](https://github.com/ziyc/drivestudio/blob/main/datasets/driving_dataset.py) 的框内LiDAR初始化思路，未整套移植框架。这是工程基线，无同场景公平OmniRe对照，不作为论文创新。原观察GT没有编辑后GT；旧静态影子问题不因增加身份自动解决。

`failure_ledger_refs=[V82-F01,V82-F02,V82-F03]`，本轮新增delta=none。首次CPU准备选了不含SciPy的解释器，partial保留；改用已有DrivingEditor解释器完成，没有安装或改共享环境，不是方法否定。
'''
(OUT/'DrivingEditor_instance_editor_report.md').write_text(body,encoding='utf-8')
repo_body = body.replace('![组件图](DrivingEditor_instance_editor_components.png)','![组件图](../autoresearch/worldsim_v82/drivingeditor_instances_20261013/r1/cpu_ready_gpu_queued/DrivingEditor_instance_editor_components.png)')
(REPO/'docs/v82/DRIVINGEDITOR_INSTANCE_EDITOR_20261013.md').write_text(repo_body,encoding='utf-8')
readme = '''# DrivingEditor v82 刚体实例基础设施

当前只验证 GMC 两个已审查动态实例；真实GT 3D框/track作为先验，局部Gaussians由整体pose移动，没有动态PVG重复运动。初始背景冻结，完整场景alpha*T可见贡献监督。入口默认validate，不占GPU；StreetCrafter及其他GPU任务优先。

完整协议、组件图和证据边界见 [实验报告](../../../docs/v82/DRIVINGEDITOR_INSTANCE_EDITOR_20261013.md)。准备、训练和编辑入口分别是 prepare_instances.py、train.py、render_edits.py；queue_gmc.py/start_queue.py只管理当前GMC300步短测，不能重复启动。

```mermaid
flowchart LR
  P[真实3D框 track LiDAR] --> L[两车局部Gaussian]
  T[真实pose 或单车编辑] --> W[刚体世界变换]
  L --> W
  S[原GN11静态背景] --> R[全场景排序与alphaT混合]
  W --> R
  R --> O[RGB 深度 可见身份贡献]
  M[真实DEVA77/82] --> C[实例可见监督]
  O --> C
```

CPU测试：`python -m pytest scripts/worldsim_v82/drivingeditor_instances/tests -q -k 'not actual_49_frame_bundles'`。真实GMC输入检查：`python check_gmc_prepared.py --prepared ACTUAL_GMC_TWO_ACTOR --output NEW_RECEIPT --cpu-threads 4`，它不验证GPU照片质量。新scene/完整mask未知时拒绝联合背景，不把缺注释当空mask。
'''
(REPO/'scripts/worldsim_v82/drivingeditor_instances/README.md').write_text(readme,encoding='utf-8')
status = f'''# 当前研究状态

更新：{now}（新加坡）。当前任务 `WS-V82-DRIVINGEDITOR-INSTANCES-20261013/r1`：用户要求v82实例编辑能力，随后收敛为只用GMC多实例场景测试。原银车/红车不新增实验；已经做好的全车辆CPU准备与旧结果保留。[协议与组件图](v82/DRIVINGEDITOR_INSTANCE_EDITOR_20261013.md) · [机器状态](v82/DRIVINGEDITOR_INSTANCE_EDITOR_20261013.json)。

已实现真实框/track分配LiDAR、局部实例Gaussian、整体刚体pose、同步协方差/局部SH、命名删/移/转/场景内替换、全场景可见贡献监督与完整checkpoint入口。车辆没有PVG重复动态，首阶段固定轨迹及背景；原SGCM只继承背景训练结果，新static→rigid克隆未做。NPS及2024活跃监督保留。代码在scripts/worldsim_v82/drivingeditor_instances；不是识别网络或无3D框方案，也不是已证明论文创新。

GMC真实输入：两个raw track分别绑定DEVA77/82，137174/160901点，共298075；49×2全部98已审查mask。laser注释有效均49，FRONT可见分别7/39。其他停车仍静态；600k数据seed仅过滤两车框后579855，实际GN11继承模型删点后的点数待GPU测。旧49窗口raw23–71和原作者窗口未知边界保留。

本地67CPU检查通过/1CUDA实测待验；本地与远端真实Python3.9均通过GMC49时刻单实例删除/平移/旋转他车参数不变与core CPU状态恢复，CUDA未初始化。完整scene GPU模型/Adam/RNG恢复、照片重建和编辑质量未实测。框可能含道路，车影及Gaussian跨框尾部未解决；不能由身份标签或CPU数学合同认证成功补景。

19:41 SGT真实启动等待controller PID20065，远端instance_editor_20261013/queue_gmc_r1。最新只读状态waiting_for_other_tasks，无GMC GPUworker。等待不导入Torch、不持共享gpu_slot.lock，5分钟稀疏检查/24小时超时。另一legacy49训练worker10164及其controller10161持共享锁是只读快照，属于其他任务；不修改、不暂停、不误杀。StreetCrafter优先；CPU最多4线程/nice10。已有每小时心跳drivingeditor-30k更新为本GMC实例任务，其他自动化不恢复；无变化安静。

下步等待空闲执行仅GMC300步GPU兼容短测，再实际49帧分层/两车编辑核验。兼容通过后同一固定GT轨迹run续至总30k上限、10k/20k/30k完整保存，不追加scene/seed/loss网格或自动开放位姿。竞争出现则保存priority_yield退出，工程异常保留资产定位；最终新增结果接入已有outputs/DrivingEditor_final_report.html，不再增多HTML入口。新GPU显存和速度尚未测，旧3090初筛不能代替新实现实测。

原C00/G10/N01/GN11各30k和GMC/红车GN11各30k共180k已完成；原18节点及真实报告保留。GN11−C00全图+0.3254dB/车体−0.1287dB，旧模块gate passed，human GN11/新scene null，完整恢复未测；新实例表示不自动套旧模块gate。此前2026-10-13 12:03:48 SGT关机请求及原自动化暂停已收口；当前主机重新在线并有其他用户任务，此新阶段没有关机、清理、恢复他人队列的授权。

SSH仅wm-vgpu-1011；新根/root/autodl-tmp/drivingeditor_repro/campaign/instance_editor_20261013，Python /root/autodl-tmp/envs/drivingeditor/bin/python。原矩阵hybrid30k_20261012与followup_scene_prep_20261013、全部checkpoint/rawFP32/输入/失败尝试保留。原静态分配与车影诊断见[F01](research_failures/entries/V82-F01.md)，环境能力迁移教训参见[F03](research_failures/entries/V82-F03.md)；本轮首次CPU解释器缺SciPy的partial保留，改已有解释器成功，无共享环境安装。failure_ledger_refs=[V82-F01,V82-F02,V82-F03]，本轮delta=none。保留scripts/run_worldsim_v72_e3_render_bridge.py无关dirty，子agent默认gpt-6.1-sol/max。
'''
(REPO/'docs/RESEARCH_STATUS.md').write_text(status,encoding='utf-8')
path=REPO/'docs/EXPERIMENTS.md'
index=path.read_text(encoding='utf-8-sig')
row='| WS-V82-DRIVINGEDITOR-INSTANCES-20261013 / r1 | 新刚体实例infra；按最新范围只测GMC双车，真实298075点/98mask，67CPU合同与目标Python3.9通过，GPU300步短测等待其他任务；照片编辑验收待测 | [协议与组件图](v82/DRIVINGEDITOR_INSTANCE_EDITOR_20261013.md) · [状态](v82/DRIVINGEDITOR_INSTANCE_EDITOR_20261013.json) · [证据](autoresearch/worldsim_v82/drivingeditor_instances_20261013/r1/cpu_ready_gpu_queued/) |\n'
if 'WS-V82-DRIVINGEDITOR-INSTANCES-20261013 / r1' not in index:
    index=index.replace('|---|---|---|\n','|---|---|---|\n'+row,1)
path.write_text(index,encoding='utf-8')
print(json.dumps({'report':str(OUT/'DrivingEditor_instance_editor_report.md'),'diagram':str(OUT/'DrivingEditor_instance_editor_components.png'),'task_state':summary['state']},ensure_ascii=False))
