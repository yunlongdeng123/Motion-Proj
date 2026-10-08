"""记录 r51 CPU 输入准入；不启动 GPU 或训练，不改 r50 分数。"""
from expand_real import O, REPO, TASK, read, dump
from pathlib import Path
from collections import Counter
import re
import shutil

m = read(O / 'manifest.json')
q = read(O / 'input_quality_review.json')
check = read(O / 'cpu_check.json')
assert check['CPU_ready'] and m['training_steps'] == 0
assert not list((O / 'inputs').glob('*/result.json'))
assert not list((O / 'inputs').glob('*/mask_result.json'))
cs = m['cases']
n = len(cs)
rejected = [r['case_id'] for r in q['cases'] if r['status'] == 'reject']
uncertain = [r['case_id'] for r in q['cases'] if r['status'] == 'uncertain']
b = dict(Counter(c['B_evidence_status'] for c in cs))
cameras = dict(Counter(c['camera'] for c in cs))
free = shutil.disk_usage(O).free / 2**30
e = REPO / 'docs/autoresearch/worldsim_v77/target_protected_20260929/r51'
e.mkdir(parents=True, exist_ok=True)
selection = {'task_id': TASK, 'run_id': 'r51', 'case_count': n, 'scene_count': n,
    'role': 'nuScenes train real DELETE development discovery; no current fine-tuning',
    'input_candidate_count': len(q['cases']), 'rejected': rejected, 'uncertain': uncertain,
    'new_vs_r50_all_38_input_reviewed_scenes': True, 'reserve_scenes': m['reserve_scenes'],
    'base': m['base'], 'seed': m['seed'], 'sampling_steps': m['sampling_steps'],
    'frames': m['frames'], 'resolution': m['resolution'], 'B_evidence': b,
    'cameras': cameras, 'cases': [{k:c[k] for k in ['case_id','scene','camera','instance_token',
        'source_id','source_manifest','source_slice','prompt_frame','median_width','median_height',
        'behind_vehicle_proxy','input_quality_review','B_evidence_status']} for c in cs],
    'failure_ledger_refs': ['V77-F02'], 'failure_ledger_delta': 'updated user triage boundary; no new generated result',
    'human_verdict': None}
for name, value in [('selection.json', selection), ('cpu_check.json', check),
                    ('input_quality_review.json', q), ('quality_exclusions.json', read(O/'quality_exclusions.json')),
                    ('r50_failure_split_user.json', read(O/'r50_failure_split_user.json')),
                    ('controller_state.json', read(O/'controller_state.json'))]:
    dump(e / name, value)
dump(e / 'sampling.json', read(O / 'sampling.json'))
dump(e / 'resources.json', {'host_alias': 'wm-vgpu-1008', 'CPU_quota': Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
    'disk_free_GiB': free, 'estimated_additional_inference_disk_GiB': 3,
    'GPU_tasks': 0, 'training_steps': 0, 'GPU_inference_not_started': True})

report = f'''# r51：新增真实 DELETE 输入，单独定位原生生成失败

任务 `{TASK}/r51`，主机 `wm-vgpu-1008`，2026-10-08。CPU已完成：从40个新scene候选准入 **{n}景/{n}例**，每景一个独立删除目标；{len(rejected)}例低质量剔除、{len(uncertain)}例不能确定不准入。SAM/DriveEditor尚未运行，训练0步。

```mermaid
flowchart LR
 X[nuScenes train 真实RGB] --> Q[三帧输入质检\n只准入2.0以上]
 Q --> M[完整SAM\n实例检查]
 M --> D[冻结官方DriveEditor]
 D --> N[原生DELETE]
 N --> W[固定写回]
 N --> F[原生结构失败\n匹配遮挡数据候选]
 W --> P[仅写回损伤\n工程队列]
 Y[真实视频Y] -. 后续监督 .-> F
```

## 先保存用户给出的失败拆分

r50用户复核将 R001/R004/R008/R028/R030/R044/R041/R062/R081 及较轻的 R037/R055 放入原生生成问题；R002/R023/R027/R029/R038/R050/R051/R060 为原生可接受、写回损伤，不用于训练diffusion。R009记为写回加重、待单独检查，不自动并入训练候选。

这是用户对已曝光开发图像的判断，不是新增推理或隐藏GT。旧AI四列分数、视频与人工记录均保留。本轮不修写回来改变对照，不把postprocess问题转成模型监督。

## 新输入和质量门槛

复用r50缓存RGB和已核对的官方SDK时刻/几何：48个尚未进入r50输入审核的场景、85个候选，按固定seed771051从40景各选一车；优先中位尺寸≥140×65px、后车布局和较大目标，原r50全窗尺寸/边距/visibility门槛仍保留。输入选择不读生成结果。

独立subagent请求5.6-sol/xhigh、未开fast，逐例打开f00/f05/f09真实原图与crop、实际B参考；按一位小数评分。目标须唯一、轮廓可辨、尺度/曝光/清晰度足以看结构；0/1分或不能确定均不进GPU。请求配置与运行模型可核验边界见JSON，不猜测实际运行身份。

准入{n}例全部≥2.0；后车证据另记 {b}，相机分布 {cameras}。B充分仅指真实可见片段，不表示隐藏完整车身有GT。后车证据不足的case可以检查再生车/背景，但不能宣称强B先验已足够。质量评分只覆盖抽帧，未判整段视频时序。

准入scene与r50全部38个输入审核scene分离，5个train隔离scene不看RGB、不训练，旧val隔离不动。来自先前数据缓存，可能与之前合成数据/官方预训练有场景重叠，因此是开发扩充，**不是最终泛化评测**。

## 下一步固定推理与分类

保持r46官方DriveEditor原权重+r21完整SAM：sam_full_v2、seed42、25steps、10帧、1024×576；无Adapter、微调、参考分支或时间模块修改。每例恢复独立原视频，只删一个actor。GPU先跑SAM并看实例身份；错误实例或空mask直接拒绝，不靠放宽门槛凑数。

准入后每例只跑一次官方DELETE，完整保存原生PNG/视频和固定写回。单帧复核先看原生：生成就错记native failure；仅最终写回错进engineering queue；两者都错分别记录，不能混成一个分数。目标/后车结构、车形再生、薄膜及向道路延伸分开备注。人工verdict留空。

只有可靠输入/实例、原生已出错的案例用于匹配后续遮挡布局和显露过程；**失败生成绝不成为训练Y，Y仍是真实视频**。本轮不造新训练数据、不训练、不追加seed/参数扫描。

## CPU验收和运行入口

{check['actual_video_decoded_frames']}帧原视频实际解码通过，10个不同真实曝光/窗、关键帧关联标注语义、场景分离和官方/SAM权重存在检查通过。CPU cgroup额度0.5核，线程设1；数据盘余约{free:.1f}GiB，预计本批推理新增不超过3GiB，无需本轮扩盘或清理。

远端run：`{O}`。入口复用r50模型/写回代码，新脚本仅管理新清单、输入准入和页面。GPU准备结束通知用户，用卡前等待开启，不后台等卡。

```bash
# GPU开启后：先SAM，审核身份后才允许delete
/root/autodl-tmp/envs/worldsim-v77-sam2/bin/python scripts/worldsim_v77/target_protected/iteration18/expand_real.py masks
/root/autodl-tmp/envs/motionproj/bin/python scripts/worldsim_v77/target_protected/iteration18/expand_real.py assets
# 保存本run mask_visual_review.json，approved/rejected覆盖全部准入case
/root/autodl-tmp/envs/driveeditor/bin/python scripts/worldsim_v77/target_protected/iteration18/expand_real.py delete
/root/autodl-tmp/envs/motionproj/bin/python scripts/worldsim_v77/target_protected/iteration18/expand_real.py assets
```

本地 `outputs/v77-real-delete-r51/index.html` 当前只展示输入，GPU列明确留空。[准入清单](../autoresearch/worldsim_v77/target_protected_20260929/r51/selection.json)、[完整输入质检](../autoresearch/worldsim_v77/target_protected_20260929/r51/input_quality_review.json)、[CPU验收](../autoresearch/worldsim_v77/target_protected_20260929/r51/cpu_check.json)、[用户r50拆分](../autoresearch/worldsim_v77/target_protected_20260929/r51/r50_failure_split_user.json)。修改前备份`/root/autodl-tmp/backups/v77_r51_cpu_20261008`。

`failure_ledger_refs: [V77-F02]`；`failure_ledger_delta: updated V77-F02（用户原生/写回归因边界）`；没有新模型失败或科学否定，人工verdict null。
'''
(REPO / 'docs/v77/REAL_DELETE_NATIVE_R51.md').write_text(report, encoding='utf-8')
status = f'''# 当前研究状态

更新：2026-10-08。分支 `research/worldsim-v7.7-target-protected-editing`，主机 `wm-vgpu-1008`。

当前唯一run `{TASK}/r51`：按用户要求再筛20–40个真实DELETE输入，先区分原生生成失败与最终写回损伤。40个新scene候选经独立5.6-sol/xhigh图像复核，{n}景/{n}个清晰单车目标准入，{len(rejected)}例低质量、{len(uncertain)}例不能确定退队列。与r50全部38个审核过的scene分离，旧5景train隔离和val隔离不动；来源仍是既有RGB缓存，用于开发而非最终泛化评测。

固定r46官方原权重+r21完整SAM，seed42/25steps/10帧/1024×576；无Adapter、训练或时间模块修改。每车独立原视频。B实际可见证据另记{b}，没有隐藏区GT。用户指定r50原生失败11例与仅写回损伤8例已分别归档，R009混合待核；旧分数/全部对照保留，写回损伤不拿去训练diffusion。

CPU准备和{check['actual_video_decoded_frames']}帧原视频解码完成，当前无GPU任务、SAM/DriveEditor各0窗、训练0步。已停在等待用户开启GPU；开卡后先SAM身份检查，再一次固定DELETE，保留四列原图/mask/原生/最终，按原生失败优先匹配造数。失败生成只能定位布局，未来Y必须真实视频，不自动微调。

本地 `outputs/v77-real-delete-r51/index.html` 当前是输入页，生成列留空。数据盘约余{free:.1f}GiB，本批预计新增≤3GiB，无需扩盘/清理；没有定时任务或电源操作。[本轮组件图、清单和入口](v77/REAL_DELETE_NATIVE_R51.md)，[V77-F02](research_failures/entries/V77-F02.md)。r50已完成的40个原生/写回结果及原有Excel评分保持归档。
'''
(REPO / 'docs/RESEARCH_STATUS.md').write_text(status, encoding='utf-8')
exp = REPO / 'docs/EXPERIMENTS.md'
text = exp.read_text(encoding='utf-8')
row = f'| {TASK} / r51 | 新40景输入复核后准入{n}景/{n}例；原生错误与写回损伤分队；CPU完成，GPU/训练0 | [报告](v77/REAL_DELETE_NATIVE_R51.md) |'
pattern = rf'^\| {TASK} / r51 \|.*$'
text = (re.sub(pattern, lambda _: row, text, flags=re.M) if re.search(pattern, text, flags=re.M)
        else text.replace('|---|---|---|\n', '|---|---|---|\n' + row + '\n', 1))
assert text.count(f'| {TASK} / r51 |') == 1
exp.write_text(text, encoding='utf-8')
card = REPO / 'docs/research_failures/entries/V77-F02.md'
text = card.read_text(encoding='utf-8')
title = '### r51 CPU：原生生成失败与写回损伤分开'
assert title not in text, '重复收口需先检查现有记录，不能追加相同段落'
text += f'''\n\n{title}

用户复核r50原生/最终两列：R001/R004/R008/R028/R030/R044/R041/R062/R081及较轻R037/R055属于原生结构问题；R002/R023/R027/R029/R038/R050/R051/R060原生可接受、写回受损，不作为diffusion训练证据；R009写回加重另核。保留旧AI评分与视频，这是用户开发图像反馈，不是新增模型运行或隐藏真值。生成与后处理失败分队，避免归因污染。

同一研究task新增r51：40个新scene候选独立复核后准入{n}景/{n}例，入口≥2.0，与r50所有38个输入审核scene分离；5景train隔离不动。CPU完成、GPU/训练0，无新科学失败或方法收益。下一步完整SAM身份核验后固定官方DELETE，原生出错才匹配遮挡布局，Y仍真实视频。[报告与组件图](../../v77/REAL_DELETE_NATIVE_R51.md)。failure_ledger_delta: updated V77-F02（用户归因边界），无新ID。
'''
card.write_text(text, encoding='utf-8')
print('RECORDED_R51_CPU', n, len(rejected), len(uncertain), b, flush=True)
