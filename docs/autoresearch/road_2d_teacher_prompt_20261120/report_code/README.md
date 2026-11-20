# 提示词配对报告

这里只准备报告源和CPU资产导出，不更改推理入口、配置或root已有index。`build_prompt_report.py`由root最终运行/覆盖本轮index。旧报告保持原值。

## 远端CPU导出（root执行）

预测完整60次、12个历史1152重放及freeze_gate通过之后：

```bash
PY=/root/autodl-tmp/envs/v42-freeinpaint/bin/python
C=/root/autodl-tmp/drivingeditor_repro/campaign/road_2d_teacher_prompt_20261120_r1
$PY "$C/report_code/export_prompt_assets.py" --predictions "$C/formal_r1/predictions.json" --gate "$C/formal_r1/freeze_gate.json" --parallel-receipt "$C/formal_r1/parallel_receipt.json" --smoke-receipt "$C/GPU_smoke_r1/receipt.json" --out "$C/report_assets_r1"
```

脚本也可直接放到其他目录，示例中的脚本路径随实际上传位置调整。`--manifest`可省略，从predictions的冻结pin读取。`--truth-config`为可选的独立clean参照；只在60完成及gate验证后读取，生成程序从不读取它。

导出保存60张1152原始raw_model PNG字节、全帧final WebP90预览、固定native ROI的raw/final无损PNG、观察与H/W/K角色视图和asset_manifest。六个ROI直接冻结上一轮窗口，并要求当前W复算仍一致；不从输出挑ROI。完整raw_native/final_native全图留远端，不为报告重复取回。WebP不用于指标或细颗粒判断。

## 本地生成

取回`report_assets_r1`到本轮`assets/`；取回统计comparison.json和per_seed.csv。下面命令在本轮目录运行：

```powershell
python report_code/build_prompt_report.py --assets assets --summary evidence/comparison.json --per-seed evidence/per_seed.csv --out report_code/rendered_r1
```

此输出仅为staging；默认图像URL是最终交付位置的`assets/`。root最后运行`--out .`，或将staging的index.html、publication.html、report_summary.json、per_seed.csv移交到本轮根目录。asset URL需改变时用`--asset-base`。

- `index.html`：本地完整交互页，默认raw110、seed101，左右prompt配对；六case、五seed、四种视图均可切换，原PNG可直接打开，支持1:1联动滚动。
- `publication.html`：轻量Git页，内嵌CSS、架构SVG和统计表，不依赖60个图像文件，不能将source PNG加入Git以使其可浏览。
- `report_summary.json`、`per_seed.csv`：完整统计与逐seed数据。没有summary时页面明确标记统计pending，不产生新prompt胜出结论。

执行开销明确区分4次smoke与60次正式，共64次，4workers各15次；实例保持开机。方法的人工作出采纳结论为accepted_method_current_cases；本轮新prompt仍未人工晋升。五seed不增加场景数；真实车下道路无GT；没有最佳seed/图片选优或三维修改。
