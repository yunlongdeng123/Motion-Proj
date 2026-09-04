# WorldSim V7.1 M24：Geometry-Locked Appearance 真渲染审计

## 目的

M23已生成冻结M8 physical centers/scales与同Actor StreetGS SH/opacity组合的attribute sidecar，但“字段可关联”不等于
现有Gaussian rasterizer能消费，也不等于视觉质量成立。M24只做一次真实渲染审计，不训练、不回写checkpoint、不把渲染
结果当作物理一致性来源。

## 冻结视图

- scene：`scene-0230`；
- Actor：`43fb20f67dbb4b149e9715f93a49e8ad`，RigidNodes index `12`，hazard construction vehicle；
- 选择理由：M5/M23既有账本中appearance support最大（32522 Gaussians），且M23 association max仅`0.426m`；这是接口
  可见性选择，不是质量选择；
- frame：完整196帧trajectory的固定中间帧`98`；
- cameras：训练配置中的`[0,1,2]`全部保留，不挑camera。

## 三个只读变体

1. `original`：冻结StreetGS checkpoint；
2. `geometry_locked`：只在内存中把目标Actor的Gaussians替换为M23的309个physical centers、isotropic scales、identity
   canonical rotations与复制的SH/opacity；trajectory和其他Actor/Background保持原对象；
3. `actor_hidden`：仅把目标Actor opacity置低，用于从`original-hidden`定义可见footprint，不作为模型候选或物理filter。

每个变体从同一checkpoint恢复后渲染全部3个camera。写出的只有PNG、mask、rows与summary；不写模型。

## 描述性评价

- rasterizer输出有限且尺寸一致；
- `original-hidden` uint8差异大于2并做2像素dilation，仅定义目标Actor视觉footprint；
- 对每个camera报告footprint pixels、original/carrier相对GT的footprint PSNR及其delta、full-image PSNR；
- 报告carrier相对original的改变像素数；零footprint camera仍保留；
- 唯一minimal decision：至少一个camera有非零目标Actor footprint，且三个变体全部成功渲染。

这些指标不进入M8/M21物理选择，不按PSNR筛view/Actor，不设置画质pass阈值。若画质下降，保留为M23 carrier的真实负
结果；不得通过调整scale/opacity/radius或删除Gaussian修图。

## 约束

- no training / fine-tuning / gradient；
- no hash / checksum / fingerprint；
- no checkpoint/background/trajectory write；
- no post-hoc image filter；hidden mask只定义评价区域；
- 单卡串行渲染，不启动多卡任务，不影响唯一AV2 downloader。

