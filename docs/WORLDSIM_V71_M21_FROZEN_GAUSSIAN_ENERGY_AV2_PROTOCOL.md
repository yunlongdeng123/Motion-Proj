# WorldSim V7.1 M21：冻结 M8 Gaussian energy 的 AV2 零样本协议

## 候选来源与偏差声明

M20事后机制诊断发现：canonical M8的children/scale不做任何新增训练，直接构成decoder-free Gaussian energy，
在已暴露nuScenes development上优于M18和M20。该候选明确是development-selected representation，不是独立source
confirmation；唯一可信确认来自冻结前从未读取quality的fresh AV2 cohort。

## 冻结模型与表示

- checkpoint：canonical M8 `20260904T202000Z__m8-temporal-frame-s71110-r2`；
- surface：immutable observed anchors + all M8 children；
- energy：`logsumexp(-||x-center||^2/(2 scale^2))`；children scale直接使用M8 GT-supervised输出，observed
  anchors固定`0.08m`；
- first return：Actor AABB内64个metric bins上的energy softmax CDF median；
- input：AV2 build-role LiDAR evidence，经冻结standardizer/M5/M8；target-role LiDAR只作最终评价；
- 无M20/M18 fine-tune、无field decoder、无AV2 calibration、无UNKNOWN mask/filter/threshold search。

这一定义是实际3D representation：energy只依赖actor-canonical coordinate与冻结Gaussian参数，同一点不因query
ray方向改变。trajectory只负责canonical-to-world刚体变换，static world独立；不输入image/semantic/hazard/time/velocity。

## 冻结数据

- cohort=`configs/worldsim_v71/av2_zero_shot_cohort_v1.json`的20个Sensor-val logs；
- downloader=`v71_download_state`的唯一现有进程；旧V7 30-log cohort无效；
- evaluator可按`.complete`逐log计算并写partial artifact，但20/20前禁止读取任何partial physical metric；
- 0-Actor log保留，不替换失败log，不启动第二下载器。

## 一次性判定

20/20后同时聚合原始baseline、M8 point surface与M21 energy：

1. energy hazard literal-early相对baseline降低至少5%；
2. energy all hit recall相对baseline下降不超过1pp；
3. M8 point Chamfer相对baseline恶化不超过1mm；
4. Actor/hazard retention均为100%。

point early/hit、clear、per-log、energy observable及M5/M7/M8/M18并列结果完整报告但不新增事后gate。若失败登记
`V71-F25`，不改anchor scale/bin/CDF阈值、cohort或checkpoint重跑。

