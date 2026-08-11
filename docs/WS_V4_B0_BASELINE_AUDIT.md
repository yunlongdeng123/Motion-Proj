# WorldSim V4 B0 Matched Baseline Audit

## 当前结论

`WS-V4-B0-MATCHED-BASELINES-01` 保持 `running`。统一评测、scene-level 统计和工程指标代码已经落地，
但当前磁盘上的 6-development-scene Tier-A baseline 资产尚不完整；首个不可变 inventory run 合法终止为
`blocked`，它是缺口诊断，不是 B0 任务终态或方法负结果。

| baseline | 当前 executable scenes | 需要 | 事实边界 |
|---|---:|---:|---|
| V3.3 frozen | 1 | 6 | scene-0230 canonical release 与其外部 base 仍可解析；其余五 scene 未物化完整链 |
| Native StreetGS | 0 | 6 | DriveStudio source/env 可执行，但历史三个 checkpoint 文件当前均已不在磁盘 |
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

## 下一动作

先提交并以干净 HEAD replay 评测/inventory；随后从公共 nuScenes archive 只抽取缺失 scene payload，补齐
6 个 StreetGS base，再在固定 official AD-GS commit 上恢复 same-split baseline。B0 只有在三种方法均达到 6/6
且统一 evaluator 生成完整 scene rows 后才可 `done`，M1 在此之前不启动。
