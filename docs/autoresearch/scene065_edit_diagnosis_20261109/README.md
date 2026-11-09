# scene-065 白色轿车删除诊断

用户要求核对是否使用最新V43编辑代码，并参考008删除银车的修复排查。报告入口为index.html，包含简单组件图、原始画面、保存的联合alpha×T、全199帧覆盖表和跨时刻实拍候选。原模型、正式算法、历史页面不变。

实际活动路径为当前共享v44_cli。公共包活动模块与当前仓库一致；最新归属修复c26d8cd5已包含。与008的完整恢复配方不同：完整目标域的native_keep清零规则未迁移，且left_keep从保留侧道路变成图表外区域，包含删除目标。因此旧语义删除的残影仍受保持约束。这是已定位的规则差异；修正收益未进行配对实验，不能宣称唯一成因或问题解决。

第119帧输入目标域11624，fit4472，keep7152，outside5296；第198帧12265、4905、7360、5373。实际删除119目标alpha0，非focus静态0.834205、其他动态0.106115、道路0.059314。alpha×T不等同颜色贡献。未知归属不等于待删车或道路，其他动态可能是合法邻车显露。

原道路图表外119/198分别只有9/10像素满足现有三来源规则；抽查的候选贴近车体轮廓、尾灯和阴影，不能称干净表面验证。当前局部平面/投影/语义规则不完整，不能外推为全视频永远无参考。道路范围内大量参考是显式邻域推断，不能混作实测。

CPU复跑：audit.py（代码）、coverage.py（199帧）、analyse_saved.py（4锚帧）、cross_time.py（119/198图表外来源）、source_figure.py（实拍图）。脚本对应本次已保存输入，写独立诊断目录，修改代码/模型0。诊断task=WAYMO-065-V43-V44-DIAGNOSIS-20261109；seed无新增随机；资源AutoDL无GPU模式CPU/单线程，无关机。

diagnose_joint.py是复用共享scene的只读CUDA诊断脚本。已通过语法检查，但本轮CUDA不可用，未执行；无RGB分层或开关实验结果。未来GPU下命令：
```
PYTHONPATH=/root/autodl-tmp/drivingeditor_repro/campaign/waymo_allframes_20261109_r1/public_api /root/autodl-tmp/envs/drivingeditor/bin/python diagnose_joint.py --manifest /root/autodl-tmp/drivingeditor_repro/campaign/waymo_val065_delete30k_20261109_r1/final_manifest.json --saved /root/autodl-tmp/drivingeditor_repro/campaign/waymo_val065_delete30k_20261109_r1/delete_white --target 10 --frames 60 119 150 198 --output /root/autodl-tmp/drivingeditor_repro/campaign/waymo_val065_delete30k_diagnosis_20261109_r1/joint_rgb_v1
```

008源实现：scene008_blue_global_road1k_20261105_r1/source/prepare_road37.py:121–125及repair37.py:46。065源：drivingeditor_dynamic_edit/v44_road.py生成views末段，v44_gate_fit.py中left及loss。旧008效果并非所有帧均无残影；只迁移已核实规则，不能把008当成完全解决的保证。

failure_ledger_refs=[V82-F06]；failure_ledger_delta=updated:V82-F06。native06530K用户通过、scene029未通过的状态不变；未代填新人工verdict。
