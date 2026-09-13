# V8.1 候选与混杂图谱

最终CPU轮 WS-V81-CPU-01 / 20260913-cpu-r2。27日志、34场景、68窗口、6120 ROI，1527几何候选；8张案例卡、49项逐图观察。模型推理=0，以下是候选/控制图，不是已验证的SOTA失效。

![Architecture components](figures/worldsim_v81/architecture.png)
![固定四格示例；保留混杂以暴露数据问题](figures/worldsim_v81/evidence_grid.png)

高重叠46项全部复核，36项排除主实验；7个非地面C00均有混杂。同日志/语义/距离的完整四格匹配组=0，不能用候选数替代干净独立对照。

![纹理重叠与CPU平面控制](figures/worldsim_v81/factor_controls.png)
![仅用于diagnostic的纹理衰减输入](figures/worldsim_v81/factor_escalation_inputs.png)
![同点数不同位置的prompt输入](figures/worldsim_v81/prompt_placement.png)

完整HTML在 /root/autodl-tmp/runs/worldsim_v81/WS-V81-CPU-01/20260913-cpu-r2/index.html；同目录 case_selection.json 固定选择规则，spot_checks.json 保存逐图观察，matched_cohorts.json 为空数组，simple_control_results.jsonl 保留有效与缺失分母。真实DONE模型结果出现后，evaluator才生成模型深度/误差/侧视对比；未生成prior-recovery或rendering/geometry mismatch图。

解释见[CPU报告](WORLDSIM_V8_1_SCIENTIFIC_REPORT.md)，限制见[方法审计](WORLDSIM_V8_1_METHOD_AUDIT.md)。人工verdict=null。
