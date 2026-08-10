# WorldSim V3.3 S4：可回滚 Spatial Delta Layer

## 结论

`WS-V33-S4-SPATIAL-DELTA-01` 已完成。最终资产状态不是新的完整 checkpoint，而是：

```text
immutable D2 base
+ S1 posterior-gated ERASE
+ S2 high-support RoadPatch INSERT_BACKGROUND
+ S3 high-support A4 INSERT_ACTOR
+ deterministic composition manifest
```

真实 DriveStudio/gsplat 验收覆盖 `base_only / erase / erase_background / actor_override / full` 五个栈、五个
冻结视角。20 次逐栈卸载后的 source render 与基座逐字节 SHA 一致，完整栈二次重放也得到同一 render SHA；D2
checkpoint 和 actor registry 前后 SHA 不变。

## Canonical 证据

- package run：
  `/root/autodl-tmp/runs/worldsim_v33/WS-V33-S4-SPATIAL-DELTA-01/20260810T221300Z__s4-package-canonical-s0-r7`
- real-render evaluation：
  `/root/autodl-tmp/runs/worldsim_v33/WS-V33-S4-SPATIAL-DELTA-01/20260810T221700Z__s4-eval-canonical-s0-r8`
- config SHA-256：`4b318a67786e576d56b6ea57d91528252fa290f0a53bd3a2f5d45dbae1c3508a`
- package manifest SHA-256：`3be8ce88764b8261740ced82a460e0109f2ce04a29c1c343c9d97ca3152bee43`
- package summary/status SHA-256：
  `cbde96004e81a6f1f0e37b7ccdd095fed364482a9754d0704052973caeda0c63` /
  `4c8332d6a9a2ed18cb3ad384218d3d43af579dd151486f39ddecde17b8d8375f`
- evaluation summary/status/decision SHA-256：
  `6f143040177cc251317328e8574ad12047803c710289159ca6eaaf5ca3c79085` /
  `87d33d951d595f93895935c2b64e8d1d9b4abd5e0f667183ecf465df0984c5e5` /
  `19e3aba6d65479701d7eef296730d974a3032c6dec13c2368ddc325547c30db9`

## 方法接口

### ERASE

ERASE 不删除任何 base tensor 行。运行时只建立新的临时 opacity Parameter，并把选中行的 logit 设为该 dtype
的最小有限值，使 `sigmoid(logit)` 精确为 `0`；上下文退出后恢复原 Parameter 对象。

S1 `hard_instance_id=13` 包含 `36,736` 个 Background 候选和 `4,525` 个 Rigid core。首个 r2 把两者全部硬擦除，
虽然目标 mask 覆盖为 `0.9997`，但目标外 L1=`0.821965`，超过冻结上限 `0.5`，因此诚实 rejected。最终合同不是
放宽门，而是使用 S1 已学得的 instance opacity：

```text
Rigid core: 全部 ERASE
Background semantic candidate: ERASE iff p(instance) >= 0.5
```

`0.5` 是概率 MAP 决策边界，不是从 S4 渲染结果搜索的阈值。最终 ERASE 为 `4,525 Rigid + 1,614 Background`
行，NPZ 保存 model code、base flat index、global Gaussian ID、instance token、selection score、policy、threshold 与
mask hash；model/index 对和 Gaussian ID 都强制唯一。

### INSERT_BACKGROUND

S2 canonical delta 同时包含 high/boundary 两个 actor。S4 high-support edit 只按冻结 `target_role=high_support`
提取其中 `25` 行，保留 source native Gaussian ID、source flat index、donor patch/chunk 和
`GENERATED_BY_PATCH_REUSE` provenance；不把 boundary 的 `79` 行混入非目标 edit。

### INSERT_ACTOR

S3 high A4 的 `99,241` 行被封装成独立 actor-local delta。每行新增连续且唯一的
`source_asset_gaussian_index` 与 `GENERATED_ACTOR` provenance。运行时不移除 Rigid core：core 已由 ERASE
将有效 opacity 归零，生成资产只追加到相同 rigid model index `5`；原 `point_ids` prefix 和其他 actor 不变。

### 固定组合与回滚

组合顺序固定为：

```text
base → ERASE → INSERT_BACKGROUND → INSERT_ACTOR → RENDER_ONLY
```

Background/Rigid 的 Parameter、Rigid `point_ids` 和 `instances_size` 都由同一上下文管理。正常退出和异常退出都恢复
进入前的对象 identity；每次真实 stack render 后再渲染 source，并比较 uint8 tensor schema + bytes SHA。

## Reference-only package

最终 package=`4,007,120 bytes`，最大文件=`3,942,422 bytes`，完整 checkpoint copy=`0`。base 目录只保存
checkpoint/registry 的绝对 reference descriptor 与 SHA：

```text
worldsim_asset/
├── base/
│   ├── checkpoint.ref.json
│   └── actor_registry.ref.json
├── deltas/
│   ├── delete_actor_13/
│   │   ├── erase.json
│   │   ├── erase_indices.npz
│   │   ├── background_patch.npz
│   │   └── manifest.json
│   └── actor_override_13/
│       ├── actor_override.npz
│       └── manifest.json
├── stacks/
│   ├── base_only.json
│   ├── erase.json
│   ├── erase_background.json
│   ├── actor_override.json
│   └── full.json
└── package_manifest.json
```

同一最终 config 下 r3/r5/r7 的 package manifest SHA 都为 `3be8ce88...ee43`，证明 package payload 与 manifest
可确定性重建；r7 被指定为 canonical，因为其 source snapshot 还包含最终 fail-terminal builder，与提交态 byte-exact。

## 真实渲染结果

冻结视角为 edit target `f091/c1`、development `f005/c0, f065/c1`、heldout confirmation
`f020/c0, f060/c1`。heldout 不参与选择或调参。

edit target：

| 指标 | 结果 | 门禁 |
|---|---:|---:|
| ERASE effect pixels | 27,000 | ≥32 |
| INSERT_BACKGROUND effect pixels | 6,663 | ≥1 |
| INSERT_ACTOR effect pixels | 14,844 | ≥32 |
| full effect pixels | 28,218 | 报告项 |
| ERASE target-mask coverage | 0.999741 | ≥0.05（与 actor 取 min） |
| actor target-mask coverage | 0.849298 | ≥0.05（与 erase 取 min） |
| full outside-target L1 uint8 | 0.225349 | ≤0.5 |

五视角 aggregate effect pixels 为 ERASE=`51,218`、background=`14,147`、actor=`28,688`、full=`54,519`；
三类 delta 均可独立开关且产生非零真实 render effect。

完整 full stack 在 target view 的首次/重放 render SHA 都为
`451ae330f58e8057472e35acbc0761a9f0027c3d34b8ca3a89a8bc6a20a50bff`。20/20 stack rollback 与额外
replay rollback 全 exact；base row deletion=`0`、erase nonzero effective opacity=`0`、duplicate insert index=`0`。

资源为 wall=`66.181 s`、peak CUDA allocated/reserved=`8,433,577,472 / 8,527,020,032 bytes`
（reserved=`8,132 MiB`），run bytes=`11,744,674`，OOM/kill delta=`0/0`；训练和 optimizer step 均为 false。

## 复现

```bash
cd /root/autodl-tmp/motion_proj
bash scripts/run_worldsim_v33_s4.sh \
  /root/autodl-tmp/runs/worldsim_v33/WS-V33-S4-SPATIAL-DELTA-01/<new-package-run> \
  /root/autodl-tmp/runs/worldsim_v33/WS-V33-S4-SPATIAL-DELTA-01/<new-eval-run>
```

launcher 会拒绝复用 run 目录或占用中的 GPU，先构建 reference-only package，再把实际 manifest SHA 传给真实
renderer evaluator，最后检查 GPU 无残留进程。

代码验证为 S4 专项 `9 passed`、V3.3/V3.2 定向联合回归 `72 passed`，并通过 `py_compile`、`bash -n` 和
`git diff --check`。

## 结论边界与下一步

- S4 证明了 scene-0230 high-support actor 的可维护空间编辑接口、真实渲染效果和精确回滚，不证明跨场景泛化；
- generated Background/Actor 仍按 S2/S3 的 provenance 和质量边界解释，不是 GT；
- boundary actor 的 generated override 仍保持 S3 `ABSTAIN_GENERATED_OVERRIDE`；
- S4 不执行通用 2D harmonization，也不授权从 heldout 重选阈值；
- 下一步只解锁 `WS-V33-S5-SEMANTIC-RENDER-01`，其核心门禁是禁止删除语义被 postprocess 重引入；R0 仍未授权。
