# 固定世界场到高斯光栅

世界场只存参考曝光下的线性RGB外观、固定表面、support和provenance；原生高斯仅查询颜色并渲染。该桥梁没有独立高斯颜色学习，也不声称恢复真实反射率/PBR或完成全场景车辆删除。

```mermaid
flowchart LR
  T[128px道路外观] --> F[world_from_uv与固定world_xyz]
  F --> Q[世界坐标查询线性RGB]
  Q --> G[固定高斯几何和opacity]
  G --> R[原生CUDA光栅]
  R --> O[顶视与斜视RGB alpha depth]
```

在repo根调用下列模块。模型训练环境为Torch2.9；这里必须使用与继承的`diff_gaussian`扩展匹配的独立环境，本轮实际为`/root/autodl-tmp/envs/drivingeditor/bin/python`（Torch2.2）。不能用训练解释器加载2.2的C++扩展。`--raster`显式指向继承原生光栅Python入口，不自动修改`sys.path`或CUDA可见性。

```bash
$PY_BRIDGE -m scripts.worldsim_v82.road_material_field.bridge.material_field --device cpu

$PY_BRIDGE -m scripts.worldsim_v82.road_material_field.bridge.run_dataset_bridge \
  --dataset "$DATA/dataset_fineguard.json" --sample "$SAMPLE_ID" \
  --texture "$NATIVE_WORLD_TEXTURE" --texture "$TRAINED_WORLD_TEXTURE" \
  --raster "$INHERITED_RASTER_PY" --output "$OUT/fresh_bridge"
```

输入清单采用corner变换：`world_from_uv@[u+.5,v+.5,0,1]`，wrapper显式转为场内部整数像素中心。固定`world_xyz`支持float32大世界坐标，容许.01 texel舍入；查询检查表面高度，不在表面的请求不获得support。颜色先sRGB→linear，再按世界位置插值；写入继承的颜色光栅前转sRGB。

CPU坐标回归覆盖大平移、旋转、非方形纹理、半像素插值、离表面拒绝和0可训练参数。真实GPU桥梁固定几何/scale/rotation/active/opacity，两个材质分支的alpha须逐值相同，输出保存RGB、alpha和depth_sum。每个run保留调用源码快照。

当前有效执行记录为`bridge/retrieved_D_eval012_final`，16384固定高斯，顶视相机物理高度3.2001057米、继承硬near=2米；旧6.8401米相机收据不是当前重放入口。完整画质、车辆遮挡与真实隐藏道路未验收，`human_verdict`保持null。包导入适配必须对有效run逐值重放，不能以新的训练或相机参数消除差异。
