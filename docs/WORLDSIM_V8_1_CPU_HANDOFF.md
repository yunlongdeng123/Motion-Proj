# V8.1 无卡阶段与 GPU 交接

用户授权范围：先做科学发现与 badcase/goodcase 可视化，不训练、不开发V8.2方法。无卡工作完成后停止，等待用户开GPU；没有后台自动恢复器。

![Architecture components](figures/worldsim_v81/architecture.png)

## 资产入口

- 首轮：`/root/autodl-tmp/runs/worldsim_v81/WS-V81-CPU-01/20260913-cpu-r1/`。
- 扩大落盘日志的冻结轮：`/root/autodl-tmp/runs/worldsim_v81/WS-V81-CPU-01/20260913-cpu-r2/`。
- 代码/轻量证据：`motion_proj/worldsim_v81`、`scripts/*worldsim_v81*`、`configs/worldsim_v81`、`docs/autoresearch/worldsim_v81`。
- 只读模型：`/root/autodl-tmp/models/worldsim_v81/`；已有VGGT：`/root/autodl-tmp/models/eas_vggt/vggt/model.safetensors`。
- 官方代码：`/root/autodl-tmp/external/worldsim_v81/{DVGT,dggt}`，VGGT沿用固定的`external/worldsim_v72/vggt`。

r2最终27日志/34场景/68窗口/6120 ROI/1527几何候选；49项视觉复核，36混杂排除，完整四格匹配组=0。r1通过几何筛查后仍发现前景栅栏混杂；r2扩大数据而不放宽标准。原始候选和排除理由都保留。报告中的控制误差是输入LiDAR平面控制，不能称为SOTA失败。

## 开卡建议

推荐 **2张、每张至少48GB显存**，DVGT-1与VGGT分别单卡推理以减少等待；**1张80GB也可顺序执行**。不需要多卡训练或分布式训练。主机至少64GB RAM、8 vCPU；当前无卡模式0.5 CPU/2GiB不适合完整模型装载。48GB建议是首轮预算估计，尚未测量本项目模型的峰值显存；首次6图batch=1测量后再决定18图oracle是否需要更大卡。

## 接续顺序

1. 读取 `docs/RESEARCH_STATUS.md` 当前节及本报告，确认r2 CPU状态。核验真实可见GPU、cgroup内存与CPU配额。
2. 使用隔离环境 `/root/autodl-tmp/envs/worldsim-v81/bin/python`。运行记录保存实际torch/CUDA版本；与DVGT官方建议torch2.8/CUDA12.8的环境差异必须在GPU合同核验解决，不能静默当成已复现。
3. 从r2输入合同按metadata顺序取一个窗口，每个模型先full6。严格官方checkpoint加载，确认native tensor shape、预处理映射、相机深度与ego坐标转换，保存峰值显存。
4. 再执行冻结的 `gpu_queue.jsonl`。full6自然候选表与sparse3/sparse2同场景诊断分开。后者只比较公共相机；针对具体ROI用 `--anchor-camera` 保留被评相机。允许新增每ROI temporal18 oracle，但不根据误差选择新ROI。
5. CPU evaluator读独立reference；DVGT用原生metric，VGGT额外保留基线定尺度前后结果。模型没有输出的相机不当作MISS，输出内部缺失才计coverage分母。
6. 稳定geometry pattern出现后，才接DGGT完整4D/渲染与成熟driving reconstruction oracle。DGGT当前core导出gs_map不等于已完成官方renderer。DriveMVS/FocusGS/VGGD仍需官方可运行代码。
7. 按[原计划](WorldSim_V81_SparseView_LowTexture_Failure_Discovery_Plan.md)做强控制、goodcase boundary与AV2确认；只有通过证据要求才改V8.2入口。

示例命令（先dry plan，无GPU不会启动模型）：

```bash
cd /root/autodl-tmp/motion_proj
PYTHONPATH=. /root/autodl-tmp/envs/worldsim-v81/bin/python scripts/run_worldsim_v81_inference.py \
  --method dvgt --variant full6 \
  --manifest /root/autodl-tmp/runs/worldsim_v81/WS-V81-CPU-01/20260913-cpu-r2/input_manifests/scene-0015_18ad5b3a.json \
  --out /root/autodl-tmp/runs/worldsim_v81/WS-V81-GPU-P2-01/dvgt/scene-0015_18ad5b3a/full6
```

实际运行时明确指定 `CUDA_VISIBLE_DEVICES=0` 并加 `--execute`；第二个模型使用另一个设备。脚本拒绝无GPU执行和覆盖已完成的同名run。首个真实窗口的三模型dry plan和官方预处理已经CPU验证；这些命令不代表GPU已运行。完整参数见r2的gpu_pilot_commands.json。

```bash
PYTHONPATH=. /root/autodl-tmp/envs/motionproj/bin/python scripts/evaluate_worldsim_v81.py \
  --atlas /root/autodl-tmp/runs/worldsim_v81/WS-V81-CPU-01/20260913-cpu-r2 \
  --prediction-root /root/autodl-tmp/runs/worldsim_v81/WS-V81-GPU-P2-01 \
  --out /root/autodl-tmp/runs/worldsim_v81/WS-V81-GPU-P2-01/evaluation
```

## 当前科学边界

模型inference=0；H1–H5未检验；SOTA badcase/goodcase均尚未发现。V8.2暂`NO_GO`只因证据未完成，不关闭研究方向。failure_ledger_delta=V81-F01/F02；人工verdict=null。

GPU前向前仍需处理实际kernel兼容性；隔离环境继承的mapanything/nuScenes-devkit依赖冲突未宣称已消除。7项检查、3模型strict meta与3个官方真实输入预处理均通过；这些不替代GPU复现。自然H1仍需补干净高重叠对照；GPU队列先服务badcase候选筛查和同场景诊断。
