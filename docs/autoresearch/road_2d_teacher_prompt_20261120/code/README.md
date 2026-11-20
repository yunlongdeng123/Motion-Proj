# 下一轮固定1152教师入口

`infer_next_round.py`默认读取上级`next_round_config.json`。最初只完成本地计划；随后按用户指令，已在真实GPU完成4次smoke及4进程的60次正式推理。实际共享生成组件为`teacher_engine.py`，调度入口为`smoke_and_workers.py`与`launch_parallel.py`。新提示词未取得稳定整体改善，活动提示词保留legacy；没有三维或历史报告修改。完整执行与结果见上级`RUNBOOK.md`。

```mermaid
flowchart LR
  A[配置与旧1152输入/预测快照] --> B[prepare 冻结60job与pins]
  B --> C[本地计划检查]
  B --> D[未来远端runtime预检 CLIP77]
  D --> E[显式execute-gpu]
  E --> F[先12次1152历史门 再完成60]
  F --> G[冻结与实际像素范围gate]
  G --> H[后续独立评分 目前未适配]
```

## 固定合同

- full1152_T0、1152×768、T0 step600、rank4注意力LoRA、DDIM20、CFG3、eta0、strength1、BF16、fp16权重；所有W/K和原生双线性硬回填语义不变。
- 六个固定case × seeds101/202/303/404/505 × legacy/material_match_v1 = 60次完整新推理，不复用旧输出，不选最佳seed，不自动进入第二轮。
- legacy×101/202共12次均引用上一轮**full1152_T0**的raw_model/raw_native/final_native三图。全部排在最前，包含在60预算内。首个或任一历史三图重放不精确即报错停止，保留现场，不改prompt/seed/方法。
- config、原manifest、原predictions和入口源码均绑定；本地源pins是Windows路径。**本地计划不能直接SCP后作为Linux执行计划**，未来远端必须用Linux来源路径重新prepare。

## 前期本地验收（历史记录）

`local_plan_r2/plan.json`为最终本地冻结计划，`local_preflight_r2.json`仅证明本地计划合同通过。`local_plan_r1`为入口收尾前的保留计划，不再使用。该前期记录的`runtime_preflight_passed=false`只描述当时状态；后续正式远端计划、真实预检与重放凭据在上级`evidence/plan.json`及worker/smoke回执中，不能用本地旧计划替代。

本地未安装transformers，缓存中未找到SD1.5 CLIP tokenizer，当时没有按单词数估算。后续真实runtime已确认旧提示词23、新提示词40 tokens，均未截断；入口仍用实际模型tokenizer，按包含BOS/EOS的完整编码计数；超过77直接停止，禁止自动截断或改词。GPU载入后也会再检查实际pipeline tokenizer。

四项针对性CPU测试覆盖：真实快照的60组合与12个1152引用、固定seed/方法及模型payload绑定、无`--execute-gpu`不创建GPU目录、构造的>77-token拒绝与三图重放不一致停止。构造token测试不是实际CLIP计数。

## 本地命令

从本次工作区根目录执行；每次prepare使用新的输出目录。

```powershell
python outputs/road_2d_teacher_defaults_20261120/code/infer_next_round.py prepare --baseline-manifest work/teacher_audit/pilot_manifest_r2.json --historical-predictions work/gpu_20261119/predictions.json --out work/next_teacher_local_check_new
python outputs/road_2d_teacher_defaults_20261120/code/infer_next_round.py preflight --manifest work/next_teacher_local_check_new/plan.json --receipt work/next_teacher_local_check_new/local_preflight.json
python outputs/road_2d_teacher_defaults_20261120/code/infer_next_round.py run --manifest work/next_teacher_local_check_new/plan.json --out work/MUST_NOT_CREATE_GPU
python -m unittest discover -s outputs/road_2d_teacher_defaults_20261120/code -p test_infer_next_round_cpu.py -v
```

`preflight`没有`--runtime`只查本地计划，不读取远端pins或加载GPU。`run`没有`--execute-gpu`也只做同一检查，out目录不会创建。

## 保留的单进程Linux入口语法

以下为单进程入口示例，重新执行须使用新输出目录。本次实际按用户要求直接真实smoke后四进程执行；独立60评分适配也已完成，使用`scoring/score_next_round.py`，不使用旧48评分器。

```bash
PY=/root/autodl-tmp/envs/v42-freeinpaint/bin/python
ROUND=/root/autodl-tmp/drivingeditor_repro/campaign/road_2d_teacher_prompt_20261120_r1
OLD=/root/autodl-tmp/drivingeditor_repro/campaign/road_2d_teacher_hf_cpu_20261119_r1
$PY "$ROUND/code/infer_next_round.py" prepare --config "$ROUND/next_round_config.json" --baseline-manifest "$OLD/prepared_r2/pilot_manifest.json" --historical-predictions "$OLD/GPU_pilot_r1/predictions.json" --out "$ROUND/prepared_r1"
$PY "$ROUND/code/infer_next_round.py" preflight --runtime --manifest "$ROUND/prepared_r1/plan.json" --receipt "$ROUND/runtime_preflight_r1.json"
$PY "$ROUND/code/infer_next_round.py" run --manifest "$ROUND/prepared_r1/plan.json" --out "$ROUND/GPU_round_r1" --execute-gpu
$PY "$ROUND/code/infer_next_round.py" freeze-gate --manifest "$ROUND/prepared_r1/plan.json" --predictions "$ROUND/GPU_round_r1/predictions.json" --receipt "$ROUND/freeze_gate_r1.json"
```

实际执行还需当时实例/环境可用，且所有冻结pins、模型与实际token检查通过。默认入口不会开机、创建服务或排队。

## 结果与评分边界

新预测schema为`road_2d_teacher_prompt_round_predictions.v1`，每job明确含`prompt_id`、seed和历史引用。`freeze-gate`核验60完成、12历史门、文件绑定、实际逆变换、W内raw硬回填、W外/K保持，并在读实际角色/RGB前核验pins；不打开GT，也不产生质量分数。

旧`score_pilot.py`硬编码48、4arms、2seeds，**不能直接评价新60结果**。本轮已完成独立60组离线诊断评分，未实现best-of-N选优模块。后续独立GT只可在所有预测完成并冻结后用于整组prompt诊断比较，不可逐图/逐seed用GT选优。真实隐藏道路没有真值。小时级后续任务仍须另做真实smoke；本地CPU合同检查不能替代GPU资格。
