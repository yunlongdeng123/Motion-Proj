# 冻结特征检索

两入口支持repo根目录下的 `python -m scripts.worldsim_v82.road_material_field.features.<入口>`，也支持直接执行文件。模型、数据与输出通过参数提供；代码不下载或提交模型。`vendor/superpoint.py`保留官方架构，唯一加载适配是支持`config.weights_path`，权重与前向不变。[原源码与权重](https://github.com/magicleap/SuperGluePretrainedNetwork/tree/master/models)及本目录许可一并保留。

```mermaid
flowchart LR
  I[同一合法原相机图] --> M[fine世界孔洞屏蔽审计]
  M --> E[冻结DINOv3共享tokens]
  E --> C[C 原图二维裁片]
  E --> D[D 相机可见加权世界场]
  C --> R[同一候选库与Top5]
  D --> R
  R --> T[128px世界外观候选]
```

DINOv3须使用支持原生架构的Transformers。执行证据复用既有Python3.12、Torch2.7.1、Transformers4.56.2特征环境；不使用LoRA或光栅的独立环境替换它。DINOv3 ViT-B/16已实际加载，不能用DINOv2或TRELLIS所需ViT-L冒名替代。权重位置显式传参：

```bash
# 在repo根执行。PY_FEATURES指向已安装匹配依赖的解释器。
$PY_FEATURES -m scripts.worldsim_v82.road_material_field.features.prepare_models \
  --dino "$DINO_WEIGHTS" --superpoint-weights "$SUPERPOINT_WEIGHTS" \
  --image "$LEGAL_ROAD_IMAGE" --output "$OUT/model_preparation.json"

$PY_FEATURES -m scripts.worldsim_v82.road_material_field.features.extract_and_retrieve \
  --dataset "$DATA/dataset_fineguard.json" --output "$OUT/fresh_retrieval" \
  --dino "$DINO_WEIGHTS" --superpoint-weights "$SUPERPOINT_WEIGHTS" \
  --run-id "$RUN_ID" --threads 4

$PY_FEATURES -m scripts.worldsim_v82.road_material_field.features.extract_and_retrieve \
  --dataset "$DATA/dataset_fineguard.json" --output "$OUT/fresh_retrieval" --package-only
```

`--world-module`可覆盖默认同包`world_feature_field.py`，`--retrieve-only`仅用于相同不可变数据修订上的缓存；`--extract-only`只提取；`--projection-density-only`不读RGB，测量原相机像素支撑。已包含`INVALIDATED.json`的输出目录被硬性拒绝继续使用，新修订必须新目录。

编码前独立投影所有sample的fine `world_xyz[hole]`，检查51实际源图每点的四个原图双线性邻像素必须全部禁止且RGB128。历史有效run为`native_controls_r3_fineguard_r1`：42000 fine texel审计0漏。旧coarse-only run作废，不复用其缓存或D参考。实际CUDA另外重做Brown投影、token采样和可见权重融合，核对独立CPU场；世界场只存实际使用点的float16特征。

A为几何/颜色/频谱；B为官方冻结SuperPoint描述子；C为同一DINOv3全帧token图上的二维裁片分箱；D只增加相机可见加权世界绑定。C不是另做crop图像前向。C/D共享帧、token预算和候选，另给共有candidate交集结果。候选只读合法clean单source input，不读supervision RGB；target仅参与排序后的评价。fine 128 texel与coarse 64 field cell单独记数，2×2重复ID不是4条独立证据。

正式4洞/10500像素的MAE为A .04687969、B .05119263、C .06861912、D .05038235；共有candidate池11/20/21/21上的C/D同趋势。D优于C但仍差于简单A约7.47%，不是训练增量或无训练性能优势。全部原生参数和ranking head更新均为0，仅为同场景伪遮挡控制，不能推断跨场景或真实隐藏道路恢复。

`summary.json`、完整`retrieval_results.json`、fine/GPU审计、`source_weight_examples.json`、16份Top5源图/对齐图和`source_target_mapping.json`保留输入、来源、位置与分母。`D_top1_reference_manifest.json`仅机械导出冻结D最小距离Top1；无输入则unknown，不从隐藏target回退。
