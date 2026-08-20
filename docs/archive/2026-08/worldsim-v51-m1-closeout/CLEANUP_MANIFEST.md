# V5.1 M1 收尾清理清单

- 执行日期：2026-08-20
- 主机模式：无卡开机，2 GiB 内存；所有审计与删除均串行执行
- 预审清单：`cleanup_plan.json`，`77,192 bytes`，SHA-256=
  `e1f2d6d553e0b5c66c8f70f726b4e0969414f49d99e066e94ad304f2c8a6b1ce`
- 执行结果：`cleanup_result.json`，`914 bytes`，SHA-256=
  `d2e7aaedd0e8af807ed3ad994609ee2056318e89eb0913f1004c1cd6b3d79482`
- 审计结果：`unsafe=0`；删除后 `remaining_cleanup_targets=0`

## 删除项

| 类别 | 精确目标数 | 文件数 | Bytes | 恢复边界 |
|---|---:|---:|---:|---|
| `*.codexbak.*` 编辑备份 | 121 | 121 | 2,211,983 | 对应原文件全部存在；候选均未被 Git 跟踪，恢复依靠 tracked original 与 Git history |
| `__pycache__` | 34 | 1,684 | 14,498,590 | Python runtime 可重建 |
| `.pytest_cache` | 1 | 5 | 153,797 | pytest 可重建 |
| **合计** | **156** | **1,810** | **16,864,370** | 不含 canonical evidence |

每个被删除文件或目录的绝对路径、bytes、文件 SHA-256 或目录 tree SHA-256、Git 跟踪检查和原文件存在性记录都保存在
`cleanup_plan.json`。删除器在执行前重新生成 inventory 并要求与该清单结构 exact 相等；任一 target 变化或
`unsafe != []` 都会拒绝删除。

## 明确保留/未触及

- `/root/autodl-tmp/motion_proj/tmp` 与 `/root/autodl-tmp/tmp` 已为空，目录保留；空 tree SHA-256 均为
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。
- `/root/autodl-tmp/mnt` 与
  `/root/autodl-tmp/third_party/trace3d-diff-id-rasterization-v51-stage-g.partial` 在预审时均不存在。
- `/root/autodl-tmp/runs/worldsim_v51/`、所有 canonical run/audit、checkpoint、KITTI ZIP、环境、正式 third-party checkout、
  source/config/test 和 Git-tracked 文档全部未删除。

## 审计过程中的非状态错误

- 两次 PowerShell→SSH inline `awk`/shell-loop 只读大小命令因引号转义失败，没有创建、修改或删除远端研究资产；这是
  `V51-F40` 已登记的跨 shell 复发。
- 首次调用假设 `/usr/bin/python3` 存在而失败；随后按仓库环境合同显式使用 `/root/miniconda3/bin/python` 完成同一预审，
  没有改变候选集合；这是 `V51-F33` 的解释器绝对路径边界复发。
