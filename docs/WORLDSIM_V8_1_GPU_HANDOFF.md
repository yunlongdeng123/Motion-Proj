# V8.1 GPU P2完成与后续入口

![Architecture components](figures/worldsim_v81_gpu/architecture.png)

状态P2_PRIMARY_DISCOVERY_COMPLETE；2×3090足够，当前推理worker均已结束，无自动恢复器，不关机。V8.2 NO_GO_PENDING_RELIABLE_FAILURE。

全部输入/原生输出/指标/日志/图：`/root/autodl-tmp/runs/worldsim_v81/WS-V81-GPU-P2-01`。正式导出使用本run的input_manifests，补充首输入相机ego时间戳；旧CPU manifest仍保留，但缺此字段时新的DVGT runner会报错，先运行repair_worldsim_v81_dvgt_export.py生成补充输入。不要重新启动已完成440项任务。

复现CPU评价：

```bash
cd /root/autodl-tmp/motion_proj
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 /root/autodl-tmp/envs/motionproj/bin/python scripts/evaluate_worldsim_v81.py \
  --atlas /root/autodl-tmp/runs/worldsim_v81/WS-V81-CPU-01/20260913-cpu-r2 \
  --prediction-root /root/autodl-tmp/runs/worldsim_v81/WS-V81-GPU-P2-01 \
  --out /root/autodl-tmp/runs/worldsim_v81/WS-V81-GPU-P2-01/evaluation_recheck --no-figures
```

新增实验使用新的task/run与输出路径，先冻结输入来源。下一步是补干净高重叠自然对照和可靠残余failure；GPU显存不是当前限制。实际runtime、完成数、缺失分母和合同修正见autoresearch/worldsim_v81/gpu_p2。
