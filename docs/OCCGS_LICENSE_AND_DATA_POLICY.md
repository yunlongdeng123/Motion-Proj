# OccGS — License 与数据使用政策

- created_at_utc: 2026-07-21T17:00Z
- 对应 gate：`G0-THIRDPARTY-00`

| 组件 | License | 使用方式 | 限制/说明 |
|---|---|---|---|
| DriveStudio (`ziyc/drivestudio` @ e59bda4) | MIT | 本地训练/渲染框架，不修改上游历史；patch 放 `motion_proj/patches/drivestudio/` | 保留版权声明 |
| gsplat v1.3.0 | Apache-2.0 | 已装入 drivestudio env | — |
| pytorch3d v0.7.5 | BSD-3 | 已装入 | — |
| nvdiffrast | NVIDIA Source Code License | 已装入（E0） | 非商用研究可用 |
| SplatAD (`carlinds/splatad`) | Apache-2.0 | 仅代码/接口审计，不安装不训练 | — |
| neurad-studio | Apache-2.0 | 仅接口参考 | — |
| nuScenes v1.0 | CC BY-NC-SA 4.0（非商用） | 本地 raw 只读 + symlink，不复制不分发 | 学术研究允许 |
| Occ3D-nuScenes gts | MIT（标注）；底层受 nuScenes 条款约束 | 暂不下载；O0 需要时仅取 3 scene 子集 | 非商用研究 |
| SegFormer cityscapes ckpt (NVlabs) | NVIDIA Source Code License（非商用） | 仅本地推理生成 sky mask | 不再分发权重 |

数据政策（承 V7 §A0.3 / §6.7）：

1. raw nuScenes 只 symlink，不复制；处理产物写 `/root/autodl-tmp/data/occgs/processed/`。
2. 任何大写盘前 `df -h /root/autodl-tmp`，保证 `avail − 预估峰值 ≥ 30 GiB`；不足先缩协议/清可重建 cache。
3. 不下载 Waymo/PandaSet/全量 Occ3D/UniScene 权重。
4. 插值(10Hz) box 一律标注 provenance=interpolated，不冒充人工 GT。
5. 合成反事实数据带完整 provenance（source scene / edit / constraint / renderer commit）。
