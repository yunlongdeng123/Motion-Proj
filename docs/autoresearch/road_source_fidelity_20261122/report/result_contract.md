# 结果JSON合同

根字段：`schema="source_fidelity_chain_result.v1"`、`run_id`、`complete=true`、`render_wh=[1920,1280]`、`geometry_unchanged=true`、`cameras_unchanged=true`、`target_migration_executed=false`、`color_encoding`、`patches`、`cases`、`checks`、`resources`。

- `patches` 必须同时包含 `patch_00`、`patch_02`。每项为 `{id,label,review}`，`review` 记录人工筛选状态及理由；不设置主参考。选项顺序仅按编号，不表示质量排序。
- `cases` 每项为 `{patch,raw,role,bbox_xyxy,crop_wh,variants}`。`bbox_xyxy` 位于1920×1280栅格，`crop_wh` 等于裁剪框的宽高。每个源块至少一个真实帧，帧数不限。
- `variants.core`、`variants.halo` 都必需，分别为核心与核心＋核验边缘。每项含 `label`、`scope`、`images`、`metrics`，可附 `display_mask`。`scope` 说明实际高斯行集合或仅显示掩码的含义。
- `images` 必须提供 `photo`、`sampled_rgb`、`loaded_rgb`、`native_render`；可附 `original_render`。每张图必须为同相机、同裁剪框的采集像素尺寸，路径相对结果JSON所在目录或为绝对路径。点RGB预览应明确采用的可见性和投影规则。
- `metrics` 为数组，每行为 `{stage,domain,score_n,RGB_MAE,RGB_bias,sigma_native_px,HF_cosine,HF_RMS_ratio}`。`RGB_bias` 可为空或三个通道值，点级预览未定义的HF值可为空。`stage` 取 `sampled_rgb`、`loaded_rgb`、`original_render`、`native_render`。渲染评价需附 `score_mask_sha256`；原源/修复源按同域比较时其掩码摘要、domain和score_n须相同。
- `checks` 每项为 `{patch,point_count,point_ids_equal,RGB_max_abs,RGB_mean_abs,geometry_unchanged,evidence}`。`evidence` 为实际核验记录路径。RGB误差按保存前后相同点ID统计；不以PNG误差替代资产数值误差。
- `resources` 为数组，每项 `{process,nativeF,nativeB,formal_updates,extra_updates}`；F/B及更新均用实际计数。`new_diffusion_calls` 与 `new_target_migration_updates` 必须为0。
- 可选 `findings` 字符串数组、`evidence` 数组 `{name,path}`、`metric_note`、`review_note`，仅写已核验结论。

本适配器只校验并展示上游真实值，不重新计算科研指标。保存/加载一致性允许真实非零误差并如实展示；相同高斯的几何冻结和相机冻结是本轮固定几何范围的必要条件。
