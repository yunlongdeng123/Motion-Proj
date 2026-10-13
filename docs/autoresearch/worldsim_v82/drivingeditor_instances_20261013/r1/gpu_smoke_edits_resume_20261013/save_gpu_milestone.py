"""Save actual GPU smoke/edit/continuation evidence without changing old results."""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/'work/drivingeditor_resume/repo_active'
WORK=ROOT/'work/instance_edit_smoke_300_r1'
SMOKE=ROOT/'work/drivingeditor_instances/gpu_smoke_milestone'
OUT=ROOT/'outputs'
EVIDENCE=REPO/'docs/autoresearch/worldsim_v82/drivingeditor_instances_20261013/r1/gpu_smoke_edits_resume_20261013'
EVIDENCE.mkdir(parents=True,exist_ok=True)
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value): Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

now=datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=8))).isoformat()
summary=read(OUT/'DrivingEditor_instance_editor_status.json')
edit=read(WORK/'edit_smoke_300_summary_r1.json')
memory=read(SMOKE/'memory_summary.json')
cuda=read(WORK/'actual_cuda_contract_r1.json')
restore=read(WORK/'receipt.json')
launch=read(WORK/'fixed30k_launch.json')
live=read(WORK/'fixed30k_status.json')
runtime=read(WORK/'fixed30k_runtime_restore.json')
reviews={name:read(WORK/name) for name in ('actor0_engineering_visual_review.json','actor1_engineering_visual_review.json','root_engineering_smoke_resume_decision_r1.json')}
summary.update(updated_sgt=now,state='GMC_FIXED_GT_INSTANCE_30000_CONTINUATION_RUNNING',
    input=dict(summary['input'],actual_warmstarted_static_points_after_GPU_exclusion=613405),
    validation=dict(summary['validation'],CPU_tests_passed=114,CPU_suite_CUDA_skip=1,
        actual_standalone_CUDA_joint_occlusion_gradient_contract='PASS',GPU_training_started=True,
        full_scene_GPU_model_Adam_RNG_restore_verified=True,
        runtime_restore_scope=restore['scope'],deterministic_next_step_parity_verified=False,
        six_edit_operations_actual_observed_frame_pairs=294,
        actual_FP32_edit_contract_passed=edit['engineering_contract']['passed'],
        photoreal_edit_quality_verified=False,human_quality_verdict=None),
    queue={'initial_smoke_queue_state':'GMC_GPU_SMOKE_COMPLETE_EDIT_VISUAL_REVIEW_PENDING',
        'initial_queue_pid':20065,'initial_queue_exited':True,'smoke_worker_pid':28170,'smoke_worker_exited':True,
        'edit_controller_pid':31353,'edit_controller_exited':True,'edits_state':'SIX_REAL_EDIT_EXPORTS_COMPLETE_REVIEW_PENDING',
        'continuation_launch':launch,'latest_run_status':live,'background_frozen':True,'pose_refinement_enabled':False,'new_SGCM_clones':False},
    GPU_smoke={'steps':300,'wall_seconds':memory['wall_seconds'],'memory':memory,
        'assessment':'300-step observation only; full new 30k/evaluation and target3090 validation pending'},
    edit_smoke={'engineering_contract':edit['engineering_contract'],
        'observed_body_mean_visible_contribution':{'pickup':edit['operations']['actor0_delete']['aggregate']['before_visible_body_mean'],
            'Jeep':edit['operations']['actor1_delete']['aggregate']['before_visible_body_mean']},
        'nonempty_mask_frames':[7,39],'empty_mask_frames':[42,10],
        'actual_contact_pages':84,'engineering_reviews':reviews,
        'limits':['300-step vehicle reconstruction not converged','dark/translucent geometry and patchy revealed road retained',
                  'frozen background removal is Gaussian-center-only and does not certify shadows or hidden surfaces',
                  'edited targets may change occlusion of other pixels without changing their assets']},
    actual_CUDA_contract=cuda,actual_runtime_restore_preflight=restore,actual_continuation_runtime_restore=runtime,
    new_evidence=str(EVIDENCE.relative_to(REPO)),
    engineering_attempts=summary.get('engineering_attempts',[])+[
        {'phase':'optional CUDA test runner','issue':'shared Python39 environment lacks pytest; stopped before Torch/GPU',
         'resolution':'standalone actual installed CUDA test; no environment install','scientific_negative':False},
        {'phase':'CPU restore guard','issue':'restored Adam aliased checkpoint input tensors',
         'resolution':'deep-copy optimizer load and regression; real GPU golden state proof passed','scientific_negative':False}])
write(OUT/'DrivingEditor_instance_editor_status.json',summary)
write(REPO/'docs/v82/DRIVINGEDITOR_INSTANCE_EDITOR_20261013.json',summary)
write(EVIDENCE/'summary.json',summary)
for directory,names in ((WORK,['edit_smoke_300_summary_r1.json','actual_cuda_contract_r1.json','receipt.json',
    'fixed30k_launch.json','fixed30k_status.json','fixed30k_runtime_restore.json','fixed30k_policy_contract.json',
    'actor0_engineering_visual_review.json','actor1_engineering_visual_review.json','root_engineering_smoke_resume_decision_r1.json']),
    (SMOKE,['status.json','effective_config.json','memory_summary.json','steps.jsonl'])):
    for name in names:
        shutil.copy2(directory/name,EVIDENCE/(('smoke_' if directory==SMOKE else '')+name))
write(EVIDENCE/'CPU_validation.json',{'command':'python -m pytest scripts/worldsim_v82/drivingeditor_instances/tests -q -k "not actual_49_frame_bundles"',
    'passed':114,'skipped':1,'scope':'CPU suite; actual CUDA proof stored separately','wall_seconds':17.32})
for name in ('launch_fixed30k.py','queue_edit_smoke.py','run_cuda_contract.py','check_full_restore.py','build_edit_contacts.py','save_gpu_milestone.py'):
    shutil.copy2(ROOT/'work/drivingeditor_instances'/name,EVIDENCE/name)

# Regenerate only the known component figure, not the old queued-state builder.
builder=(ROOT/'work/drivingeditor_instances/save_infra_report.py').read_text(encoding='utf-8').split('report = read(PREP')[0]
builder=builder.replace('CPU 合同已通过 · GPU 测试排队，不抢其他任务','GPU 300 步与单车编辑合同通过 · 固定轨迹继续至总 30k')
exec(compile(builder,'components_only','exec'),{'__file__':str(ROOT/'work/drivingeditor_instances/save_infra_report.py')})
for name in ('DrivingEditor_instance_editor_components.png','DrivingEditor_instance_editor_components.svg'):
    shutil.copy2(OUT/name,EVIDENCE/name)

pid=launch['pid']; step=live.get('step'); peak=memory['nvidia_smi']['device_peak_used_bytes']/2**30
body=f'''# v82 DrivingEditor：GMC 双实例固定轨迹基线

更新：{now}。仅 GMC scene0008 raw23–71；其他两场景没有新增实验。真实300步GPU短测完成，两车各删除、局部前移0.5m和局部yaw30度覆盖49观察帧，共294对。工程合同和实际恢复通过，已从step300续到总30000（新增29700），worker PID{pid}，当前快照step{step}；10k/20k/30k完整保存。StreetCrafter及其他GPU任务优先，CPU最多4线程/nice10，竞争出现保存priority_yield后退出，无自动抢占续跑。

![组件图](DrivingEditor_instance_editor_components.png)

车辆使用静态局部Gaussian与真实整体刚体轨迹，协方差和局部SH方向同步；没有车辆PVG重复运动。两条原始track明确绑定DEVA77/82，137174/160901个真实LiDAR点，共298075，98/98真实可见mask；有效轨迹各49帧，FRONT可见7/39。600k准备seed仅排除这两框后579855；实际GN11继承背景经Gaussian中心框内删除后613405点，两种计数不能混用。停车仍在静态背景。

逐实例监督来自完整场景的同一次深度排序alpha*T，而不是孤立实例覆盖度。实际安装CUDA测试得到可见union0.392、孤立union0.980，alpha梯度有限并通过身份贡献求和。首阶段静态/环境冻结、真实轨迹固定、拓扑固定，SGCM只继承原GN11背景结果，没有新static→rigid克隆；NPS原公式及RGB/静态RGB/深度衰减保留，不宣称绝对静态或新SGCM迁移完成。

294组FP32独立CPU复核有限性、逐帧GT/合成/static/vehicles配对、独立static前后逐位相同和删除目标贡献精确0。非编辑实例Gaussian世界位置、旋转、尺度、外观与opacity在实际GPU执行中逐位不变；它的可见像素仍可因合法遮挡变化。旧GMC49帧GT PNG复核最大byte差0。84张接触页保留全图和真实mask40pxpadding共同ROI，空mask帧用全图。

300步真实图像仍有车体偏黑、半透明、纹理和轮廓不足，删除后道路浅灰模糊patch及暗残影。工程视觉审核保留这些坏例并允许按原预算训练，不能称照片编辑质量通过。背景Gaussian中心移除不处理跨框尾部、车影或遮挡后未观测表面；没有编辑后GT、重照明或OmniRe公平对照。human_quality_verdict仍null，不把300步与旧30k硬比，不套旧四臂1dB判据。

114项CPU检查通过，CPU suite的CUDA项默认跳过；另一个独立真实CUDA前后向测试通过。完整场景模型/Adam与sampler/Python/NumPy/TorchCPU/allCUDA RNG已从真实300步checkpoint在GPU逐字段逐值恢复核验；续训入口也在第一步更新前重复核验。此证明的范围是保存状态恢复，不证明下一步CUDA数值确定性或未来所有节点可恢复；旧checkpoint认证字段保留False。旧300步没有raw配置快照，策略根据其真实effective_config与现有reference重建，历史配置内容认证边界明确保留；后续checkpoint保存实际解析policy，锁定seed、损失、LR horizon、冻结/位姿开关及拓扑。

300步总wall59.442秒，末300步训练吞吐约6.95步/秒；PyTorch allocated/reserved峰2.105/3.508GiB，进程/设备采样峰3.840/3.850GiB、60采样0errors。它只描述短测，不能认证新30k全程或3090容量/速度；采样峰是下界，仍须完整训练、编辑/评价及目标机验证。各编辑操作memory_summary原样保留远端。此新阶段不关机、不改其他任务、不改共享环境。

主输出：`runs/gmc_two_actor_fixed30k_r1`；启动凭据：`queue_gmc30k_r1/launch.json`；先前短测：`queue_gmc_r1/gpu_smoke`；六组编辑：`edits_smoke_300_r1`。每小时检查保持开启，无变化安静。最终真实收敛结果接入已有DrivingEditor_final_report.html，不新增HTML入口。旧六组180k、旧报告/离线包和所有失败资产保留。
'''
(OUT/'DrivingEditor_instance_editor_report.md').write_text(body,encoding='utf-8')
repo_body=body.replace('![组件图](DrivingEditor_instance_editor_components.png)',
    '![组件图](../autoresearch/worldsim_v82/drivingeditor_instances_20261013/r1/gpu_smoke_edits_resume_20261013/DrivingEditor_instance_editor_components.png)')
(REPO/'docs/v82/DRIVINGEDITOR_INSTANCE_EDITOR_20261013.md').write_text(repo_body,encoding='utf-8')
(EVIDENCE/'report.md').write_text(body,encoding='utf-8')
status=f'''# 当前研究状态

更新：{now}。当前任务WS-V82-DRIVINGEDITOR-INSTANCES-20261013/r1，仅GMC两动态实例。300步GPU兼容、六项×49观察帧编辑合同、真实完整模型/Adam/RNG恢复通过；从同checkpoint续至总30k，PID{pid}，当前证据step{step}，10k/20k/30k完整保存。

[协议与组件图](v82/DRIVINGEDITOR_INSTANCE_EDITOR_20261013.md) · [状态](v82/DRIVINGEDITOR_INSTANCE_EDITOR_20261013.json) · [新GPU证据](autoresearch/worldsim_v82/drivingeditor_instances_20261013/r1/gpu_smoke_edits_resume_20261013/)。旧CPU输入/合同证据保留。114CPU检查通过，实际CUDA联合遮挡/梯度另行通过。真实98mask/298075实例点，GN11继承静态删两框中心后613405；停车保持静态。固定背景/环境、固定GT轨迹、固定拓扑，车辆无PVG重复运动，不自动开放pose或joint background；旧SGCM仅继承背景，没有新克隆。NPS原公式保留。

真实300步车体偏黑透明、道路patch与残影均保留，工程支持继续原预算，不代表收敛质量。人工评价null；编辑后GT不存在。恢复证明仅首次更新前模型/Adam/RNG精确值，不认证下一步确定性或全部未来checkpoint；旧300配置没有raw快照的边界保留，今后policy锁定。GPU短测设备采样峰{peak:.3f}GiB是下界，新30k和目标3090仍待验证。

每小时心跳ACTIVE，StreetCrafter及其他GPU任务优先，不共跑；CPU≤4/nice10。新竞争任务出现保存priority_yield退出，无自动抢资源续训。仅SSH wm-vgpu-1011；新根/root/autodl-tmp/drivingeditor_repro/campaign/instance_editor_20261013。默认CPUvalidate，禁止重复启动已退出的300队列或当前worker。

银车/红车不新增实验。旧六组180k与统一报告全部保留，旧关机路线已结束，此新阶段绝不关机、清理、暂停或终止他人任务。保留scripts/run_worldsim_v72_e3_render_bridge.py无关dirty，子agent默认gpt-6.1-sol/max。
'''
(REPO/'docs/RESEARCH_STATUS.md').write_text(status,encoding='utf-8')
index=REPO/'docs/EXPERIMENTS.md'
text=index.read_text(encoding='utf-8-sig')
lines=text.splitlines()
for i,line in enumerate(lines):
    if line.startswith('| WS-V82-DRIVINGEDITOR-INSTANCES-20261013 / r1 |'):
        lines[i]='| WS-V82-DRIVINGEDITOR-INSTANCES-20261013 / r1 | GMC双刚体实例；300步GPU/6×49编辑/完整GPU模型AdamRNG恢复通过，114CPU检查；从同step300续至总30k，固定背景轨迹；未收敛透明车体/道路patch保留 | [协议与组件图](v82/DRIVINGEDITOR_INSTANCE_EDITOR_20261013.md) · [状态](v82/DRIVINGEDITOR_INSTANCE_EDITOR_20261013.json) · [GPU证据](autoresearch/worldsim_v82/drivingeditor_instances_20261013/r1/gpu_smoke_edits_resume_20261013/) |'
index.write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'state':summary['state'],'pid':pid,'step':step},ensure_ascii=False))
