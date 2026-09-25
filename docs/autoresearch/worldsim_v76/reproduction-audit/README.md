# P0 复现审计证据

完整分析和 architecture components 图见 [报告](../../../v76/P0_REPRODUCTION_AUDIT.md)。本目录保存两项诊断、工程错误与可复现修复；大输入和 checkpoint 留在原 AutoDL run。

- `colmap_audit.json`：138,944 点及实际初始化/visibility 审计，含逐图计数和修复后的305图像映射验证。
- `official_test_16k_metrics.json`：原官方时间 split 的75视图、每相机指标、acc与白像素诊断；旧P0权重仍包含V76-F01，数值未因源码修复而重命名为修复后效果。
- `official_test_16k_frame20.png`：固定帧20，五相机GT、render、acc；不是最佳样例选择。
- `fix_colmap_view_mapping.patch.gz`：对VAD-GS上游commit `77e27686d84b64be4643d518cea77b34bf9718bf` 的局部补丁，包含helper与3项回归测试；gzip保留上游补丁上下文的原始空白。
- `audit_colmap.py`、`evaluate_official_test.py`：复制到VAD-GS `script/v76/` 使用。审计脚本的源数据和相机范围为本次P0固定输入。
- `INVALIDATED.json`：旧训练主动停止的原因；`run_p0.sh` 已加此标记检查，防止误resume。

修复代码验证：

```bash
cd /root/autodl-tmp/external/worldsim_v75/VAD-GS
# 补丁已部署；在干净上游树复现时先 gzip -dc 解压，再 git apply --check 和 git apply。
/root/autodl-tmp/envs/vadgs-v76/bin/python script/v76/test_colmap_view_mapping.py
/root/autodl-tmp/envs/vadgs-v76/bin/python -m py_compile lib/utils/drivestudio_utils.py script/v76/evaluate_official_test.py
git diff --check
```

诊断命令（不重新训练）：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= \
/root/autodl-tmp/envs/vadgs-v76/bin/python script/v76/audit_colmap.py \
  --run /root/autodl-tmp/runs/v76_ego_view/VADGS-P0-000 \
  --output /root/autodl-tmp/runs/v76_ego_view/VADGS-P0-000/v76_diagnosis/colmap_audit.json
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/root/autodl-tmp/envs/vadgs-v76/bin/python script/v76/evaluate_official_test.py \
  --config configs/v76/nuscenes_000_train.yaml --iteration 16000 \
  --output-dir /root/autodl-tmp/runs/v76_ego_view/VADGS-P0-000/v76_official_test_16k
```

`failure_ledger_refs: [V76-F01]`；`failure_ledger_delta: V76-F01`。旧P0全部权重仅作工程诊断；下一次有效训练必须新建run、重建初始化，不加载HUGSIM或旧P0权重。
