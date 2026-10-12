"""Assemble CPU audit evidence; reads scratch evidence only."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
events = json.loads((ROOT / "events_loss_trends_cpu.json").read_text())
losses = json.loads((ROOT / "loss_formula_gradient_cpu.json").read_text())
depth = json.loads((ROOT / "sparse_depth_collisions_cpu.json").read_text())
candidate = json.loads((ROOT / "candidates/candidates_tests_remote_cpu.json").read_text())
checkpoint_path = ROOT.parents[1] / "vgpu_1011/checkpoint_030000_cpu_audit_20261011T154802Z/remote_stdout.json"
checkpoint = json.loads(checkpoint_path.read_text())
BASE = "/root/autodl-tmp/drivingeditor_repro"
FROZEN = BASE + "/campaign/resume_vgpu_baseline_20261011/frozen_repository"
AUDIT = BASE + "/campaign/vgpu_bugfix_20261012/loss_audit"

reset_rows = []
for iteration, data in events["reset_summaries"].items():
    row = {"reset_iteration": int(iteration)}
    for name in ("pre_100", "post_1_10", "post_1_100", "post_101_300"):
        row[name + "_psnr_mean"] = data["train/psnr"][name]["mean"]
    row["exact_psnr"] = data["train/psnr"]["exact"]
    reset_rows.append(row)

windows = []
for name, data in events["window_summaries"].items():
    windows.append({
        "window": name, "limits_inclusive": events["windows"][name],
        "psnr_mean": data["train/psnr"]["mean"],
        "global_rgb_mean": data["train/loss_paper_global_rgb"]["mean"],
        "total_loss_mean": data["train/loss"]["mean"],
    })

classification = [
    {"item": "Sparse depth duplicate-pixel order dependence", "class": "confirmed inherited projection bug",
     "paper": "Eq16, PDF7/printed1702", "source": "utils/camera_utils.py117-133",
     "evidence": "Point-order reversal changes identical point-set target and reverses Eq16 gradient; real3frames8.09-11.73% populated pixels last-write farther than nearest.",
     "limitation": "Nearest positive finite z is a stated geometry policy; paper omits pixel collision implementation. Branch-collapse causality unproven.", "candidate": "candidates/nearest_depth.patch"},
    {"item": "Loss cannot uniquely identify static/dynamic layer assignment", "class": "confirmed implemented-objective degeneracy; reproduction-domain ambiguity",
     "paper": "Eq7,9-10,13-16, PDF5-7/printed1700-1702", "source": "utils/paper_losses.py27-95;train.py337-400",
     "evidence": "Actual loss/SSIM/compose functions give both proper split and static-empty/dynamic-full conceptual constructions1.250062504e-10 total.",
     "limitation": "Per-pixel construction is not a proof of finite3D Gaussian realizability, optimizer trajectory, or author intent.", "candidate": "candidates/opacity_outside.patch, explicitly experimental"},
    {"item": "Static opacity reset every3000 after densification ends", "class": "confirmed inherited scheduling instability; observed reset discontinuity",
     "paper": "Optimization details PDF8/printed1703; no reset timing specified", "source": "train.py738-766;scene/gaussian_model.py409-412",
     "evidence": "At30k scalar forward34.513dB;next step8.591dB; analogous12/15/18/21/24/27/33k pattern. Saved30k eval10.4835dB after reset.",
     "limitation": "Explains immediate global checkpoint discontinuity, not earlier20k branch allocation failure. No isolated GPU intervention yet."},
    {"item": "Static branch temporal geometry versus paper time-invariant G_s", "class": "confirmed executable semantics mismatch; exact intended enforcement ambiguous",
     "paper": "Eq4 PDF5/printed1700 vs Eq1-3 PDF4/printed1699", "source": "gaussian_renderer/__init__.py259-273;scene/gaussian_model.py147-178,269-298",
     "evidence": "Static xyz uses SHM(timestamp), marginal opacity varies; xyz/t/beta/rawvelocity trainable. Largebeta can hide velocityfeature despite0.31831-model-unit SHM amplitude.",
     "limitation": "Actual checkpoint drift magnitude must be measured; paper does not state hard-freeze versus soft enforcement."},
    {"item": "Dynamic env present in train and absent in exported actor channel", "class": "confirmed train/export diagnostic mismatch; shared env train behavior matches paper",
     "paper": "Eq6 and dynamic-render prose PDF7/printed1702", "source": "gaussian_renderer/__init__.py115-120;scene/cameras.py94-106",
     "evidence": "render_d receives residual sharedenv only is_training=True; train uses masked env-filled dynamicRGB; eval component channel is raw actor.",
     "limitation": "Not a global composite-eval defect. Camera jitter is inherited training sampling and separate diagnostic variable."},
    {"item": "Depth normalization / harmonic900sky convention", "class": "normalization verified; author depth convention ambiguous",
     "paper": "Eq16 PDF7/printed1702", "source": "train.py360-373,399",
     "evidence": "Already divides by compositealpha; subsequent harmonic sky blend adds total-opacity gradient. World-to-model scale.1 is consistent for rendered/LiDAR targets.",
     "limitation": "Cannot certify author intended normalized surface versus environment-filled depth."},
    {"item": "NPS channel/self-target/arithmetic/gradient and dynamic xyz LR", "class": "no confirmed bug in tested formulas/source",
     "paper": "Eq9 PDF6/printed1701, training details PDF8/printed1703", "source": "train.py333-335,394;utils/paper_losses.py41-74;scene/gaussian_model.py891-901",
     "evidence": "Independent loops and doublegradcheck pass;feature uses channels1:4 and alpha normalization;dynamicxyz1.6e-3→1.6e-6 exactly matches paper.",
     "limitation": "Exact checkerboard offsets/distances/numericallambda values and hardclamp boundary stabilization are not disclosed."},
]

payload = {
    "scope": "Read-only CPU audit plus isolated reviewable candidates; no GPU training, formal source/config/data/checkpoint edits",
    "host_alias": "wm-vgpu-1011", "actual_source": FROZEN, "remote_audit": AUDIT,
    "snapshot_sha256": hashlib.sha256((ROOT / "code_snapshot.tar.gz").read_bytes()).hexdigest(),
    "source_pins": losses["source_pins"], "event_sha256": events["event_sha256"],
    "paper_source": "User supplied15-page DrivingEditor TIP35(2026)PDF; last numbered equation16, no17",
    "released_author_head_verified": {"repository": "https://github.com/WangXu-xxx/DrivingEditor", "commit": "560d57c3825ed2bdbc6a6602e0011ad51e8da60e", "commit_date": "2024-12-29T17:38:12+08:00", "verification": "git ls-remote refs/heads/main + official publicREADME; not final2026 loss source"},
    "classification": classification,
    "formula_checks_passed": losses["all_formula_checks_passed"], "candidate_remote_tests_passed": candidate["all_passed"],
    "candidate_test_count": len(candidate["tests"]), "window_means": windows, "reset_windows": reset_rows,
    "reset_scalar_timing": events["forward_timing"],
    "final_scalar_step": max(p[0] for p in events["scalar_points_step_value_walltime"]["train/psnr"]),
    "eval_scalars": {k: [p[:2] for p in v] for k, v in events["eval_scalars"].items()},
    "weighted_pre30k": events["weighted_components"]["29900_30000_pre_reset_and_save"],
    "collision_records": depth["records"],
    "saved30k_adam": {k: checkpoint["checkpoint_components"][k]["adam_saved_state_validation"] for k in ("static", "dynamic")},
    "optimizer_evidence_file": str(checkpoint_path),
    "objective_counterexample": candidate["full_objective_layer_degeneracy_counterexample"],
    "outside_gradient_example": candidate["opacity_two_pixel_evidence"],
    "next_experiment": {"name": "complement_background_domain", "status": "CPU preparation only; no launch", "one_variable": "new outsideopacity objective coefficient=.05, unchangedforegroundEq13", "baseline": "true originalfrozen30k→30300harness from component agent", "gates": ["baseline completion", "root review", "explicit GPU handoff"], "unchanged": ["30k4originalcheckpoint files", "savedRNG/sampler", "50khorizon", "paperforegroundEq13", "allpapercoefficients", "depth", "SGCM", "reset"]},
}
(ROOT / "handoff_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

lines = [
    "# DrivingEditor 2026 损失与梯度审计交接（CPU）",
    "",
    "确定发现稀疏深度的点顺序覆盖错误。完整目标还存在逐像素分支分配退化：静态分支为空、动态分支覆盖整幅背景可以获得与正确分离相同的目标值。30k 全局图像骤降主要与保存前静态 opacity reset 的瞬时断点一致；这不能解释20k已出现的道路分支混入。现有NPS数值、通道、邻居与梯度检查通过。",
    "",
    f"审计只使用实际执行的 `{FROZEN}`，与child的`code_snapshot.tar.gz`一致；没有训练、GPU或正式源/配置/checkpoint修改。snapshot SHA256：`{payload['snapshot_sha256']}`。完整JSON及候选位于 `{AUDIT}`，本地同名`work/vgpu_bugfix_20261012/loss_audit/`。",
    "",
    "```mermaid",
    "flowchart LR",
    '  Input["相机与时间"] --> S["静态 Gaussian：SHM / α / v"]',
    '  Input --> D["动态 Gaussian：SHM / α"]',
    '  Input --> E["共享环境图"]',
    '  S --> R["静态 / 动态 / 全局渲染"]',
    '  D --> R',
    '  E --> R',
    '  GT["GT RGB、动态/天空 mask、LiDAR"] --> L["全局RGB+深度；静态Sky+NPS；前景RGB+Opacity"]',
    '  R --> L',
    '  L --> Out["Adam → 分支与组合图像"]',
    "```",
    "",
    "论文核对范围是Eq7–16：用户提供PDF最后一个编号是16，未找到Eq17。Eq8属于SGCM，由独立SGCM审计负责。公开作者仓库HEAD经`git ls-remote`核实仍为[2024 commit560d57](https://github.com/WangXu-xxx/DrivingEditor/tree/560d57c3825ed2bdbc6a6602e0011ad51e8da60e)，不能以它的旧全图MSE代替2026论文公式。",
    "",
    "| 论文页/式 | 实际冻结实现 | 判断 |",
    "|---|---|---|",
    "| PDF5/1700 Eq7 | `paper_losses.py27-31`：-mean(o_sky log(1-o_sky))，仅天空，稳定clamp | 公式/解析梯度通过；alpha达到clamp上界后的零梯度是稳定化选择 |",
    "| PDF6/1701 Eq9 | `paper_losses.py41-74`；`train.py333-335,394`：静态feature1:4 / alpha，四轴相邻，两个棋盘格parity，距离权重，sum | 独立loop与gradcheck通过，无self-target或通道错误；精确采样偏移/边界/距离定义未公开 |",
    "| PDF6/1701 Eq10 | `compose_paper_loss90-95`：sky+NPS | 论文未要求额外staticRGB；不能把旧staticRGB回填叫作已证修复 |",
    "| PDF7/1702 Eq11–12 | `gaussian_model.py147-153`及renderer187-194：damped average v与SHM时间位移 | 动态时间移动与公式相符；具体扰动采样未公开 |",
    "| PDF7/1702 Eq13 | `paper_losses.py34-38`：动态前景mask ×(1-a)log(a)/HW | 公式未显示domain；当前选择按前景prose，mask外梯度为0。字面全图同式会把背景alpha推向1，也不能解决背景归属 |",
    "| PDF7/1702 Eq14–15 | `train.py384-397`：L1+.2SSIM，动态RGB预先乘mask，前景opacity+.photo | 系数.2匹配；全图零填充SSIM/面积归一化是解释选择，不能认证ROI作者实现 |",
    "| PDF7/1702 Eq16 | `train.py360-373,399`与`paper_losses.py78-87`：alpha归一化后harmonic900sky混合；有效稀疏逆深度/HW | reciprocal/有效域/归一化CPU通过；harmonic约定未公开；camera_utils.py133的duplicate last-write确定有问题 |",
    "",
    "| 分类 | 证据与含义 | 尚未证明 |",
    "|---|---|---|",
]
for item in classification:
    lines.append(f"| {item['class']}：{item['item']} | {item['evidence']} (`{item['source']}`) | {item['limitation']} |")
lines += [
    "",
    "`loss_formula_gradient_cpu.json`包含Eq7/13解析导数、Eq9独立循环与双精度gradcheck、Eq16独立参考。NPS常量非零速度场loss/grad均0；单Gaussian ray的alpha尺度改变几乎不改变NPS（alpha梯度3.55e-15）。这是一致性loss的零空间，不是遗漏某个已给出的强制论文项。NPS还可通过增大beta缩小damped velocity，示例beta=.2的导数-1.5163；实际训练是否利用该路径尚需checkpoint分布验证。",
    "",
    "`candidates/candidates_tests_remote_cpu.json`给出完整目标反例：4×4图、中心2×2 mask、GT/env/颜色=.5、两个LiDAR采样20，proper split与static空/dynamic满的globalRGB、masked dynamicRGB、sky、NPS、depth均0，前景opacity同为2.500125e-9，实际compose总值均1.2500625e-10。harmonic深度也计入构造。它证明已实现目标的逐像素不可识别性，不证明有限3DGaussian在全部视角可实现、真实优化路径或作者原意。",
    "",
    "静态语义需特别注意：论文Eq4称G_s time-invariant；实际静态renderer仍使用SHM(timestamp)及Gaussian时间marginal，xyz/t/beta/rawvelocity皆可训练。原代码`get_inst_velocity`实际是经beta衰减的average velocity，而SHM振幅使用raw velocity。CPU例rawv=10,beta=10,T=.2的vmap约1.39e-10，但SHM振幅0.31831模型单位（本场景scale=.1，相当3.1831m）。这是可检验的语义差异，尚未测定真实checkpoint漂移量；不能直接假定作者要求硬冻结。",
    "",
    "学习率没有发现动态xyz乘10的尾值错误：静态1.6e-4×spatialscale→1.6e-6×scale，动态1.6e-3→1.6e-6，50k指数horizon，符合论文PDF8/1703。两分支velocity恒定.001×scale、beta.002、time.0008→.000008，时间与速度均未被静态硬冻结。先前只读30k checkpoint验证保存LR：static xyz1.009531751e-5、dynamic xyz2.535829108e-5；两者time5.047658756e-5、velocity.001、beta.002。Adam条件step不必等global iteration。",
    "",
    "训练dynamicRGB的真实输入包含共享env残差，论文PDF7/1702明确dynamic遵循Eq6并共享env，因此env本身不能定为bug。train.py341还对预测render_d乘mask；Eq14文字只说rendered dynamic image与gt_d，是否需要未mask预测全图的黑背景监督，因共享env也有歧义。outside实验隔离缺失背景域的因果方向，即便改善也不能证明Eq13本应包含该项，更不能未经核对把取消prediction mask称为确定修复。导出eval actor通道缺env，无法直接以该通道与训练loss值比较。训练env rays的像素jitter与eval中心光线差别属独立诊断项。rasterizer backward源码有feature→alpha/feature梯度，但本CPU审计未执行CUDA kernel，不能认证实测所有参数梯度。",
    "",
    "| inclusive window | PSNR均值 | globalRGB均值 | 总loss均值 |",
    "|---|---:|---:|---:|",
]
for w in windows:
    lines.append(f"| {w['limits_inclusive'][0]}–{w['limits_inclusive'][1]} | {w['psnr_mean']:.6f} | {w['global_rgb_mean']:.9f} | {w['total_loss_mean']:.9f} |")
lines += [
    "",
    "| reset | reset前100步 | 后1–10步 | 后1–100步 | 后101–300步 |",
    "|---:|---:|---:|---:|---:|",
]
for r in reset_rows:
    lines.append(f"| {r['reset_iteration']} | {r['pre_100_psnr_mean']:.6f} | {r['post_1_10_psnr_mean']:.6f} | {r['post_1_100_psnr_mean']:.6f} | {r['post_101_300_psnr_mean']:.6f} |")
lines += [
    "",
    "窗口为原event所有逐步scalar，不是抽样。reset前100步为[k-99,k]，上表需求窗口为[k-100,k]，因此均值存在微小差别。step30000训练forwardPSNR34.513126，30001=8.590572；20k保存eval35.248520，30k保存eval10.483502。train scalar发生在reset前，eval/checkpoint在reset及optimizer后；最终scalar35917。train.py764-766在densify截止条件外只重置static opacity（dynamic不重置），并替换参数、清零opacity Adam moments。跨这些周期重复断点与其后恢复排除了“30k前已全局长期发散”的简单说法，不能排除20k已有分支退化。",
    "",
    "29900–30000加权分量：globalRGB.012870597（94.56%），dynamicRGB.000163947（1.20%），sky.000022024（.162%），NPS.000044115（.324%），dynamicopacity.000000386（.00284%），depth.000509991（3.75%）。各分量标量比例不等于Gaussian参数梯度比例；未据此宣称NPS无效或应回退旧loss。",
    "",
    "| frame | 有效depth pixels | last-write比nearest远 | 占有效pixel | duplicate z跨度中位数/最大m | λ=.1逆depth target差异尺度 |",
    "|---|---:|---:|---:|---:|---:|",
]
for d in depth["records"]:
    lines.append(f"| {d['frame']} | {d['populated_depth_pixels']} | {d['last_write_not_nearest_pixels']} | {100*d['last_write_not_nearest_fraction_depth_pixels']:.4f}% | {d['collision_span_m_quantiles']['0.5']:.3f}/{d['collision_span_m_quantiles']['1.0']:.3f} | {d['weighted_eq16_difference_scale_lambda_0_1']:.9f} |")
lines += [
    "",
    "深度检查复用相同49pose的冻结PCA、scale=.1、480×320 pixels；frame10点顺序反转改变2593个pixel，其中last-farther2518个（2497在mask外）。两点z5/10、renderz7反转顺序导致Eq16导数-0.0204081→+0.0204081。表中最后列是target逆深度改变的尺度，不是实际render loss改变测量；最近可见点policy需明确标注，不能据此证明它是分支退化主因。",
    "",
    "隔离候选均可`git apply --check`于本地snapshot和实际remote frozen，通过CPU测试但没有应用正式源。`nearest_depth.patch`用finite positive z的`np.minimum.at`消除碰撞顺序依赖，测试collision/24permutations/noncollision/empty/zero/NaN/Inf/behind-camera/实际loadCam函数。",
    "",
    "`opacity_outside.patch`保留Eq13前景，新增可选实验背景项：L_out=-mean[(1-M)·a·log((1-a+eps)/(1+eps))]，eps=1e-4，默认λ=0，实验λ=.05。它沿Eq7的背景零opacity方向补足域，而不是旧100倍全图MSE；高alpha没有hardclamp造成的上界死区。alpha[.8,.8],mask[1,0]原foreground梯度[-.2365718,0]，新增outside梯度[0,2.8035195]。λ=.05只改变mask外，反例collapsed总值增至.330170063，proper保持1.2500625e-10。该系数背景面积可能大于前景，惩罚规模并不与原foreground相同；这是待检验的假设、不是认证作者公式。",
    "",
    "最小实验顺序：先完成真实原frozen30k→30300 baseline；随后仅在独立source应用outside项，新增显式hypothesis flag和λ=.05，沿用相同30k四文件/RNG/sampler/50khorizon；同5camera节点和终点49frame，完整四件checkpoint与实际状态。独立guard仅放行该新增flag/weight及两文件指定diff，保持正式resumehelper严格；baseline完成、rootreview及GPU交接之前不launch。主要判据是mask外dynamicalpha、static道路恢复、actor内质量与global质量相对baseline变化，不能仅以PSNR更高宣布忠实复现。",
    "",
    "后续逐项独立候选：nearestdepth只改变目标归约；static时不随时间渲染可测试time-invariant语义；resetstop仅测试周期断点；NPS是否应采用其他feature/距离及λ必须单独调查作者资料。现在不得把depth/SGCM/reset与outside约束一起改。当前完整15项候选测试在localtorch2.12CPU和remotetorch2.2.2CPU均通过；公式参考检查全部通过。",
    "",
    "证据清单：`handoff_summary.json`、`events_loss_trends_cpu.json`（所有逐步点与完整窗口）、`loss_formula_gradient_cpu.json`、`sparse_depth_collisions_cpu.json`、`candidates/manifest.json`、两候选patch、`candidates/candidates_tests_remote_cpu.json`、CPU脚本。所有可证结论与未知项在JSON中分别标明。",
]
(ROOT / "HANDOFF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"handoff": str(ROOT / "HANDOFF.md"), "summary": str(ROOT / "handoff_summary.json"), "classification_count": len(classification), "candidate_tests": len(candidate["tests"]), "all_passed": candidate["all_passed"]}, ensure_ascii=False))
