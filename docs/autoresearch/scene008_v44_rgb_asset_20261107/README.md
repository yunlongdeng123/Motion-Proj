# v44：当前车体与 RCR-1 去蓝背景接入

代码接线与 CPU 预检阶段，GPU 当前不可用；新组合未渲染，未确认画质。

![组件](architecture.svg)

RCR-1 的已保存 native10K（补全图 RGB×1）与自身 fresh road1K/gates，配当前 C2/固定RGBA/SurfaceLOD。原 11×9 色场仅作 frozen-transfer，不称为新场景重新标定。保持同一车体，原生 a6 物理移除，保留/删除仅开关 a37；原生合成不等于 MOG-2 不透明首交。

历史 raw137 银亮片净增亮主要是 static RGB，环境×T 含车形；去蓝没有解决银亮片。新组合 GPU 后按固定 ROI 和真实分组 RGB 复查，不用亮度后选子集充当固定分母。HELD9 已参与内部开发，不是独立泛化。

执行：见 `RUN_COMMANDS.txt`。第一步实际 GPU pilot 四帧，工程检查后再 full49。无新训练、无模型权重更改，无自动电源操作。现有 v43/RCR-1/车体报告不覆盖。

failure_ledger_refs=[]；failure_ledger_delta=none；human_verdict=null。
