# WorldSim V4 B0 Matched Baseline Audit

## 当前结论

`WS-V4-B0-MATCHED-BASELINES-01` 保持 `running`。统一评测、scene-level 统计和工程指标代码已经落地；
6 个 development scenes 的 DriveStudio 输入与 StreetGS sky masks 已全部物化，scene-0048 的 100-step
训练 profile 已通过。当前仍缺 6-scene formal StreetGS、V3.3 replay 与 AD-GS same-split 资产，因此 B0
尚未闭环；所有 blocked run 都是独立诊断，不是方法负结果。

| baseline | 当前 executable scenes | 需要 | 事实边界 |
|---|---:|---:|---|
| V3.3 frozen | 1 | 6 | scene-0230 canonical release 与其外部 base 仍可解析；其余五 scene 未物化完整链 |
| Native StreetGS | 0 formal / 1 profile | 6 formal | 6-scene 数据齐备；scene-0048 profile100 通过，尚未登记 30k formal checkpoint |
| AD-GS | 0 | 6 | 6-scene historical metrics 仍在；source/env/checkpoint 当前均缺失，历史数值不算 executable |

## 冻结协议

- development scenes：`scene-0230/0242/0255/0048/0994/0139`，精确来自 D0 cohort；
- 同 scene、同 `sample_index mod 5` train/development/heldout；sensor `1600×900`、source downscale=2、
  model/metric `800×450`；不得读取 test quality；
- 图像主指标固定 `PSNR/SSIM/LPIPS-Alex`，区域固定 global/static/actor/boundary/edit_roi；
- 无 GT 或空区域返回 `undefined`，禁止以生成结果自身作 GT；
- 主统计单位固定为 scene；failed/blocked/abstain 保留 denominator；
- 工程指标由 raw timing/count/resource rows 派生，不允许手填 success/yield/retry 比率。

## Diagnostic run

- run：`/root/autodl-tmp/runs/worldsim_v4/WS-V4-B0-MATCHED-BASELINES-01/20260811T090951Z__b0-inventory-diagnostic-s0-r1`
- terminal：`blocked / matched_baseline_assets_incomplete`
- matrix/summary/manifest/status/fingerprint/inventory SHA-256：
  `8f5d31e65f0710cfdc5751aada3381ee48eed42736d59403227b628fecbf9eea` /
  `b25a9c35b726f9aa1a5b75ed3ccf84998e8703b99a05f9c548fb8619a0df8e93` /
  `ce7e437fea0951292e5ef2b69d4d84d1ffff767e80b176b8724cc87e4f93ba49` /
  `1c1e702f38fe7860d798ccb8bc7fa4217b903e411407e1d46af93549de77fadb` /
  `3bbbaf960e4072b846028f5293db2ee6ee9bc9f7b5bef047ff4d0df878f7145b` /
  `40a322c6ee8190823bba3cbfa7339fc4ffd66b00d492d5cacc57aa0c26e31a0b`
- run size：`53,158 bytes`；training/model inference/test quality 均为 `false`。

## 六场景数据与 sky-mask 里程碑

- 公共 nuScenes archive 缺失三场景的 canonical extraction run：
  `20260811T092132Z__b0-data-extract-missing3-s0-r3`，terminal=`done`，`5,264/5,264` members，
  fingerprint=`c51f41622ab4023ccaea656836576855d717be2b93c2d4a8692eba2562379e38`；只扫描官方 10 个 tar shard，
  不下载数据、不读取 test quality；
- preprocess canonical runs：scene-0048/0139/0994 分别为
  `20260811T102622Z__b0-preprocess-scene0048-s0-r5`、
  `20260811T102857Z__b0-preprocess-scene0139-s0-r6`、
  `20260811T103142Z__b0-preprocess-scene0994-s0-r7`，三者均 `done`；r4 因上游 `_10Hz` 输出目录合同
  未对齐而 blocked，未覆盖；
- canonical processed root=
  `/root/autodl-tmp/data/worldsim_v4/drivestudio_processed_10Hz/trainval`；scene index
  `045/110/179/191/204/752` 均精确为 `1,176 RGB / 196 LiDAR / 588 sky masks`；
- SegFormer 固定 `nvidia/segformer-b5-finetuned-cityscapes-1024-1024@2c6f153e...`。远端官方网络恢复 r10
  因 `Errno 101` blocked；本机从官方固定 URL 下载后按 bytes/SHA256 exact 校验并传入远端 staging，r11
  `20260811T110746Z__sky-model-restore-staging-exact-s0-r11` 以
  `restore_mode=codex_local_official_staging_exact / network_attempted=false` 完成，fingerprint=
  `15c200fdd8bd6eeb5999d639076b3c5d0d38c76523f1c7f187181ec7b4790bfd`；
- sky-mask runs：r12 因 preprocess 预建空 target 被旧合同拒绝并保留 blocked；修复只接受空目录、任何已有
  mask/partial 仍 fail-closed。scene-0048/0139/0994 canonical r13/r14/r15 均完成 `588/588`，fingerprint=
  `c35a9dbe...c3eb / 5f6a2fe7...c702 / 455dcb20...e6c7`，峰值 GPU 均 `2,876 MiB`，全程离线且
  test quality 未读。

## StreetGS profile 门

- r9 已真实启动训练，但在 iteration 之前因 scene-0048 缺 `sky_masks/000_0.png` blocked；GPU 子进程已退出；
- 修复数据链后 canonical profile=
  `20260811T111810Z__streetgs-scene0048-profile100-s0-r16`，terminal=`done`；100 steps / `90.8965 s`，
  checkpoint=`340,298,602 bytes / SHA256 446297b897024b326def46c66f05e7e30cf2736dcc767789180652e8498203af`，
  Gaussian=`Background 1,074,426 / RigidNodes 4,746`，峰值 GPU=`9,004 MiB`、cgroup=`10,867,355,648 bytes`，
  fingerprint=`62ebeb1ef11dc6785e1031f1555188c97e3e17d426101a316584abf3765c2eda`；
- profile 只证明 30k 训练链和资源前置可执行，不计入 formal coverage，不登记图像质量改进。

## 下一动作

按固定 scene 顺序串行完成 6 个 StreetGS 30k formal checkpoint，并逐场登记 bytes/SHA/资源；随后恢复固定 official
AD-GS commit 的 same-split baseline，并补齐 V3.3 六场景 replay。B0 只有在三种方法均达到 6/6 且统一 evaluator
生成完整 scene rows 后才可 `done`，M1 在此之前不启动。
